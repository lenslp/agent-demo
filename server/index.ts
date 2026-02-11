
import Koa from 'koa';
import Router from 'koa-router';
import bodyParser from 'koa-bodyparser';
import cors from '@koa/cors';
import { generateText, tool, stepCountIs } from 'ai';
import { openai } from '@ai-sdk/openai';
import { z } from 'zod';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { Client } from '@modelcontextprotocol/sdk/client';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio';

dotenv.config();

const app = new Koa();
const router = new Router();
const port = 3001;

// Define the root directory for file operations (parent of 'server' directory)
const PROJECT_ROOT = path.resolve(__dirname, '..');

app.use(cors());
app.use(bodyParser());

// --- MCP Integration (Start) ---

interface MCPStdioConfig {
    type?: 'stdio';
    command: string;
    args?: string[];
    env?: Record<string, string>;
    envFile?: string;
}

interface MCPRemoteConfig {
    type?: 'sse' | 'http';
    url: string;
    headers?: Record<string, string>;
    auth?: {
        CLIENT_ID: string;
        CLIENT_SECRET?: string;
        scopes?: string[];
    };
}

type MCPServerConfig = MCPStdioConfig | MCPRemoteConfig;

interface MCPConfig {
    mcpServers: Record<string, MCPServerConfig>;
}

/**
 * Convert MCP tool schema to Zod schema
 */
function mcpSchemaToZod(mcpSchema: any): z.ZodTypeAny {
    if (!mcpSchema || !mcpSchema.type) {
        return z.any();
    }

    switch (mcpSchema.type) {
        case 'string':
            return z.string();
        case 'number':
            return z.number();
        case 'integer':
            return z.number().int();
        case 'boolean':
            return z.boolean();
        case 'array':
            return z.array(mcpSchemaToZod(mcpSchema.items || {}));
        case 'object':
            if (mcpSchema.properties) {
                const zodProps: Record<string, z.ZodTypeAny> = {};
                for (const [key, value] of Object.entries(mcpSchema.properties)) {
                    zodProps[key] = mcpSchemaToZod(value);
                }
                return z.object(zodProps);
            }
            return z.record(z.any());
        default:
            return z.any();
    }
}

/**
 * Create an AI SDK tool from an MCP tool definition
 */
function createToolFromMCP(mcpTool: any, mcpClient: Client) {
    const inputSchema = mcpTool.inputSchema 
        ? mcpSchemaToZod(mcpTool.inputSchema)
        : z.object({});

    return tool({
        description: mcpTool.description || `MCP tool: ${mcpTool.name}`,
        inputSchema: inputSchema as z.ZodObject<any>,
        execute: async (args: any) => {
            try {
                const result = await mcpClient.callTool({
                    name: mcpTool.name,
                    arguments: args || {},
                });
                
                if (result.isError) {
                    return { error: result.content?.[0]?.text || 'Unknown error' };
                }
                
                // Extract content from MCP result
                const content = result.content?.[0]?.text || JSON.stringify(result.content);
                return { result: content };
            } catch (error: any) {
                return { error: error.message || 'Failed to call MCP tool' };
            }
        },
    });
}

/**
 * Interpolate configuration values (like Cursor does)
 * Supports: ${env:NAME}, ${userHome}, ${workspaceFolder}, ${workspaceFolderBasename}
 */
function interpolateConfig(value: string, workspaceFolder: string): string {
    return value
        .replace(/\$\{env:([^}]+)\}/g, (_, name) => process.env[name] || '')
        .replace(/\$\{userHome\}/g, os.homedir())
        .replace(/\$\{workspaceFolder\}/g, workspaceFolder)
        .replace(/\$\{workspaceFolderBasename\}/g, path.basename(workspaceFolder))
        .replace(/\$\{pathSeparator\}/g, path.sep)
        .replace(/\$\{\/\}/g, path.sep);
}

/**
 * Interpolate all string values in a config object recursively
 */
function interpolateConfigObject(obj: any, workspaceFolder: string): any {
    if (typeof obj === 'string') {
        return interpolateConfig(obj, workspaceFolder);
    } else if (Array.isArray(obj)) {
        return obj.map(item => interpolateConfigObject(item, workspaceFolder));
    } else if (obj && typeof obj === 'object') {
        const result: any = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = interpolateConfigObject(value, workspaceFolder);
        }
        return result;
    }
    return obj;
}

/**
 * Load environment variables from envFile
 */
function loadEnvFile(envFilePath: string, workspaceFolder: string): Record<string, string> {
    const resolvedPath = path.isAbsolute(envFilePath) 
        ? envFilePath 
        : path.join(workspaceFolder, envFilePath);
    
    if (!fs.existsSync(resolvedPath)) {
        console.warn(`⚠️  Env file not found: ${resolvedPath}`);
        return {};
    }
    
    const env: Record<string, string> = {};
    const content = fs.readFileSync(resolvedPath, 'utf-8');
    
    for (const line of content.split('\n')) {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#')) {
            const match = trimmed.match(/^([^=]+)=(.*)$/);
            if (match) {
                const key = match[1].trim();
                const value = match[2].trim().replace(/^["']|["']$/g, '');
                env[key] = value;
            }
        }
    }
    
    return env;
}

/**
 * Read MCP configuration from mcp.json file (Cursor format)
 */
function readMCPConfig(): MCPConfig | null {
    // Try project-level config first: .cursor/mcp.json
    const projectConfigPath = path.join(PROJECT_ROOT, '.cursor', 'mcp.json');
    if (fs.existsSync(projectConfigPath)) {
        try {
            const content = fs.readFileSync(projectConfigPath, 'utf-8');
            return JSON.parse(content);
        } catch (error: any) {
            console.error(`❌ Failed to read project MCP config: ${error.message}`);
        }
    }
    
    // Try global config: ~/.cursor/mcp.json
    const globalConfigPath = path.join(os.homedir(), '.cursor', 'mcp.json');
    if (fs.existsSync(globalConfigPath)) {
        try {
            const content = fs.readFileSync(globalConfigPath, 'utf-8');
            return JSON.parse(content);
        } catch (error: any) {
            console.error(`❌ Failed to read global MCP config: ${error.message}`);
        }
    }
    
    return null;
}

/**
 * Initialize MCP clients and load tools
 */
async function initializeMCPTools(): Promise<Record<string, any>> {
    const mcpTools: Record<string, any> = {};
    
    // Read MCP configuration from mcp.json (Cursor format)
    const mcpConfig = readMCPConfig();
    if (!mcpConfig || !mcpConfig.mcpServers) {
        console.log('ℹ️  No MCP servers configured. Create .cursor/mcp.json or ~/.cursor/mcp.json to enable MCP tools.');
        return mcpTools;
    }

    try {
        const mcpServers = mcpConfig.mcpServers;
        
        for (const [serverName, config] of Object.entries(mcpServers)) {
            try {
                console.log(`🔌 Connecting to MCP server: ${serverName}...`);
                
                // Interpolate config values
                const interpolatedConfig = interpolateConfigObject(config, PROJECT_ROOT);
                
                // Determine transport type
                const transportType = interpolatedConfig.type || 
                    (interpolatedConfig.url ? 'sse' : 'stdio');
                
                if (transportType === 'stdio') {
                    // STDIO transport
                    const stdioConfig = interpolatedConfig as MCPStdioConfig;
                    
                    // Load env file if specified
                    let env = { ...stdioConfig.env } || {};
                    if (stdioConfig.envFile) {
                        const envFileVars = loadEnvFile(stdioConfig.envFile, PROJECT_ROOT);
                        env = { ...envFileVars, ...env };
                    }
                    
                    const transport = new StdioClientTransport({
                        command: stdioConfig.command,
                        args: stdioConfig.args || [],
                        env: env,
                    });
                    
                    const client = new Client({
                        name: `agent-demo-${serverName}`,
                        version: '1.0.0',
                    }, {
                        capabilities: {
                            tools: {},
                        },
                    });

                    await client.connect(transport);
                    
                    // List available tools
                    const toolsResult = await client.listTools();
                    
                    if (toolsResult.tools && toolsResult.tools.length > 0) {
                        console.log(`  ✓ Found ${toolsResult.tools.length} tools from ${serverName}`);
                        
                        // Convert each MCP tool to AI SDK tool
                        for (const mcpTool of toolsResult.tools) {
                            const toolName = `${serverName}_${mcpTool.name}`;
                            mcpTools[toolName] = createToolFromMCP(mcpTool, client);
                            console.log(`    - ${toolName}`);
                        }
                    } else {
                        console.log(`  ⚠️  No tools found in ${serverName}`);
                    }
                    
                } else {
                    // SSE/HTTP transport (not yet implemented in this example)
                    console.warn(`  ⚠️  Remote MCP servers (SSE/HTTP) are not yet supported. Skipping ${serverName}.`);
                }
                
            } catch (error: any) {
                console.error(`❌ Failed to connect to MCP server ${serverName}:`, error.message);
            }
        }
        
        if (Object.keys(mcpTools).length > 0) {
            console.log(`✅ Loaded ${Object.keys(mcpTools).length} MCP tools`);
        }
        
    } catch (error: any) {
        console.error('❌ Failed to initialize MCP tools:', error.message);
    }
    
    return mcpTools;
}

// Initialize MCP tools (will be populated asynchronously)
let mcpTools: Record<string, any> = {};
initializeMCPTools().then(tools => {
    mcpTools = tools;
    console.log('🚀 MCP tools initialization complete');
}).catch(error => {
    console.error('❌ MCP tools initialization failed:', error);
});

// --- MCP Integration (End) ---

// --- Tool Definitions (Start) ---

const calculateTool = tool({
    description: 'A tool for performing basic math calculations.',
    inputSchema: z.object({
        expression: z.string(),
    }),
    execute: async ({ expression }) => {
        try {
            // eslint-disable-next-line no-eval
            const result = eval(expression);
            return { result };
        } catch (error) {
            return { error: 'Invalid expression' };
        }
    },
});

const getCurrentTimeTool = tool({
    description: 'Get the current local time.',
    inputSchema: z.object({}),
    execute: async () => {
        return { time: new Date().toLocaleString() };
    },
});

const readFileTool = tool({
    description: 'Read the contents of a file in the project directory.',
    inputSchema: z.object({
        filename: z.string(),
    }),
    execute: async ({ filename }) => {
        try {
            const filePath = path.isAbsolute(filename) ? filename : path.join(PROJECT_ROOT, filename);
            if (!fs.existsSync(filePath)) {
                return { error: `File ${filename} not found.` };
            }
            const content = fs.readFileSync(filePath, 'utf-8');
            return { content };
        } catch (error: any) {
            return { error: error.message };
        }
    },
});

const writeFileTool = tool({
    description: 'Write content to a file at the specified path. If the file exists, it will be overwritten.',
    inputSchema: z.object({
        filepath: z.string().describe('The path to the file to write. Relative to project root or absolute.'),
        content: z.string().describe('The content to write to the file.'),
    }),
    execute: async ({ filepath, content }) => {
        try {
            const filePath = path.isAbsolute(filepath) ? filepath : path.join(PROJECT_ROOT, filepath);
            const dir = path.dirname(filePath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            fs.writeFileSync(filePath, content, 'utf-8');
            return { success: true, message: `File written to ${filePath}` };
        } catch (error: any) {
            return { error: error.message };
        }
    },
});

const deleteFileTool = tool({
    description: 'Delete a file at the specified path. The file must exist and be within the project directory.',
    inputSchema: z.object({
        filepath: z.string().describe('The path to the file to delete. Relative to project root or absolute.'),
    }),
    execute: async ({ filepath }) => {
        try {
            const filePath = path.isAbsolute(filepath) ? filepath : path.join(PROJECT_ROOT, filepath);
            
            // Security check: ensure the file is within the project root
            const resolvedPath = path.resolve(filePath);
            const resolvedRoot = path.resolve(PROJECT_ROOT);
            if (!resolvedPath.startsWith(resolvedRoot)) {
                return { error: `Cannot delete file outside project root: ${filepath}` };
            }
            
            if (!fs.existsSync(filePath)) {
                return { error: `File ${filepath} not found.` };
            }
            
            // Check if it's a directory
            const stats = fs.statSync(filePath);
            if (stats.isDirectory()) {
                return { error: `${filepath} is a directory. Use deleteDirectory tool instead.` };
            }
            
            fs.unlinkSync(filePath);
            return { success: true, message: `File ${filepath} deleted successfully.` };
        } catch (error: any) {
            return { error: error.message };
        }
    },
});

// Base tools (always available)
const baseTools = {
    calculate: calculateTool,
    getCurrentTime: getCurrentTimeTool,
    readFile: readFileTool,
    writeFile: writeFileTool,
    deleteFile: deleteFileTool,
};

// Combined tools (base + MCP tools)
function getTools() {
    return {
        ...baseTools,
        ...mcpTools,
    };
}

// --- Tool Definitions (End) ---

router.post('/api/chat', async (ctx) => {
    try {
        const { messages } = ctx.request.body as { messages: any[] };

        if (!messages || !Array.isArray(messages)) {
            ctx.status = 400;
            ctx.body = { error: 'Messages array is required' };
            return;
        }

        console.log('Received messages:', messages.length);

        const availableTools = getTools();
        const toolNames = Object.keys(availableTools);
        
        // Check if user message contains action verbs that require tool usage
        const lastMessage = messages[messages.length - 1];
        const userMessage = typeof lastMessage?.content === 'string' 
            ? lastMessage.content.toLowerCase() 
            : '';
        const requiresAction = /(提交|commit|push|执行|执行|做|完成|帮我|请|git|add|status)/.test(userMessage);
        
        const { text, toolCalls, response } = await generateText({
            model: openai('gpt-4o-mini'),
            system: `You are an ACTION-ORIENTED AI Agent named DemoAgent. 
          
          CRITICAL RULES:
          1. When the user asks you to DO something (提交代码, commit, push, etc.), you MUST USE TOOLS immediately.
          2. DO NOT ask for confirmation or additional information UNLESS absolutely necessary.
          3. If you need information to complete a task, USE TOOLS to get it first.
          4. Execute actions proactively - don't just describe what you would do.
          
          Available tools: ${toolNames.join(', ')}
          
          For git operations:
          - Use git-mcp tools (git-mcp_*) to check status, add files, commit, and push
          - If git-mcp tools are available, use them directly without asking
          - For "提交代码" or "commit code", you should: check status → add files → commit → push
          
          For file operations: use readFile, writeFile, deleteFile tools.
          
          Current working directory: ${PROJECT_ROOT}
          User home directory: ${os.homedir()}
          
          WORKFLOW FOR ACTION REQUESTS:
          1. Identify which tools are needed
          2. Call the tools immediately to gather information or perform actions
          3. Continue calling tools until the task is complete
          4. Report results after completion
          
          REMEMBER: You are an executor, not a consultant. When asked to do something, DO IT using tools.`,
            messages: messages,
            tools: availableTools,
            toolChoice: requiresAction ? 'required' : 'auto', // Force tool usage for action requests
            stopWhen: stepCountIs(10), // Increased from 5 to allow more tool calls
        });

        if (toolCalls && toolCalls.length > 0) {
            console.log('\n🛠️ Tools used:', toolCalls.map(c => c.toolName).join(', '));
        }

        console.log(`\n✨ Agent response generated`);

        ctx.body = {
            messages: response.messages,
            text: text
        };

    } catch (error: any) {
        console.error('Error in /api/chat:', error);
        ctx.status = 500;
        ctx.body = { error: error.message || 'Internal Server Error' };
    }
});

app.use(router.routes()).use(router.allowedMethods());

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
    console.log(`Project root: ${PROJECT_ROOT}`);
});
