
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
import { exec } from 'child_process';
import { promisify } from 'util';
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
                console.log(`🔧 Executing MCP tool: ${mcpTool.name} with args:`, JSON.stringify(args));
                
                const result = await mcpClient.callTool({
                    name: mcpTool.name,
                    arguments: args || {},
                });
                
                if (result.isError) {
                    const errorMsg = result.content?.[0]?.text || 'Unknown error';
                    console.log(`❌ Tool ${mcpTool.name} failed:`, errorMsg);
                    return { 
                        success: false,
                        error: errorMsg,
                        toolName: mcpTool.name
                    };
                }
                
                // Extract content from MCP result - handle multiple content types
                let content = '';
                if (result.content && result.content.length > 0) {
                    const contentParts = result.content.map((item: any) => {
                        if (item.type === 'text') return item.text;
                        if (item.type === 'resource') return `[Resource: ${item.resource?.uri || 'unknown'}]`;
                        return JSON.stringify(item);
                    });
                    content = contentParts.join('\n');
                } else {
                    content = JSON.stringify(result);
                }
                
                console.log(`✅ Tool ${mcpTool.name} succeeded. Result length: ${content.length} chars`);
                
                return { 
                    success: true,
                    result: content,
                    toolName: mcpTool.name,
                    message: `Tool ${mcpTool.name} executed successfully`
                };
            } catch (error: any) {
                console.error(`❌ Tool ${mcpTool.name} exception:`, error.message);
                return { 
                    success: false,
                    error: error.message || 'Failed to call MCP tool',
                    toolName: mcpTool.name
                };
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

// --- Skills Integration (Start) ---

interface CodeBlock {
    language: string;
    code: string;
    index: number;
}

interface SkillInfo {
    name: string;
    description: string;
    path: string;
    content: string;
    codeBlocks: CodeBlock[];
    hasExecutableScripts: boolean;
}

/**
 * Load available Cursor skills
 */
function loadSkills(): Record<string, SkillInfo> {
    const skills: Record<string, SkillInfo> = {};
    const skillsDir = path.join(os.homedir(), '.cursor', 'skills-cursor');
    
    if (!fs.existsSync(skillsDir)) {
        console.log('ℹ️  Skills directory not found:', skillsDir);
        return skills;
    }
    
    try {
        const skillDirs = fs.readdirSync(skillsDir, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);
        
        for (const skillDir of skillDirs) {
            const skillPath = path.join(skillsDir, skillDir, 'SKILL.md');
            if (fs.existsSync(skillPath)) {
                try {
                    const content = fs.readFileSync(skillPath, 'utf-8');
                    // Parse YAML frontmatter
                    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
                    let description = `Cursor skill: ${skillDir}`;
                    let name = skillDir;
                    
                    if (frontmatterMatch) {
                        const frontmatter = frontmatterMatch[1];
                        const nameMatch = frontmatter.match(/^name:\s*(.+)$/m);
                        const descMatch = frontmatter.match(/^description:\s*(.+)$/m);
                        if (nameMatch) name = nameMatch[1].trim();
                        if (descMatch) description = descMatch[1].trim();
                    }
                    
                    // Extract code blocks from content
                    const codeBlocks: CodeBlock[] = [];
                    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
                    let match;
                    let index = 0;
                    const executableLanguages = ['bash', 'sh', 'shell', 'python', 'python3', 'node', 'javascript', 'js', 'typescript', 'ts'];
                    
                    while ((match = codeBlockRegex.exec(content)) !== null) {
                        const language = (match[1] || '').toLowerCase();
                        const code = match[2].trim();
                        if (code) {
                            codeBlocks.push({
                                language,
                                code,
                                index: index++
                            });
                        }
                    }
                    
                    const hasExecutableScripts = codeBlocks.some(block => 
                        executableLanguages.includes(block.language) || block.language === ''
                    );
                    
                    skills[`skill_${skillDir}`] = {
                        name,
                        description,
                        path: skillPath,
                        content: content.replace(/^---\n[\s\S]*?\n---\n/, '').trim(), // Remove frontmatter
                        codeBlocks,
                        hasExecutableScripts
                    };
                    
                    const scriptInfo = hasExecutableScripts 
                        ? ` (${codeBlocks.filter(b => executableLanguages.includes(b.language) || b.language === '').length} executable scripts)`
                        : '';
                    console.log(`📚 Loaded skill: ${name} (${skillDir})${scriptInfo}`);
                } catch (error: any) {
                    console.warn(`⚠️  Failed to load skill ${skillDir}:`, error.message);
                }
            }
        }
        
        if (Object.keys(skills).length > 0) {
            console.log(`✅ Loaded ${Object.keys(skills).length} skills`);
        }
    } catch (error: any) {
        console.error('❌ Failed to load skills:', error.message);
    }
    
    return skills;
}

/**
 * Execute a script from a code block
 */
async function executeScript(language: string, code: string, workingDir: string = PROJECT_ROOT): Promise<{ success: boolean; output: string; error?: string }> {
    const execAsync = promisify(exec);
    
    try {
        let command = '';
        
        switch (language.toLowerCase()) {
            case 'bash':
            case 'sh':
            case 'shell':
            case '': // Default to bash if no language specified
                command = code;
                break;
            case 'python':
            case 'python3':
                // Write code to temp file and execute
                const tempFile = path.join(os.tmpdir(), `skill_${Date.now()}_${Math.random().toString(36).substring(7)}.py`);
                fs.writeFileSync(tempFile, code);
                command = `python3 "${tempFile}"`;
                // Clean up temp file after execution
                setTimeout(() => {
                    try { fs.unlinkSync(tempFile); } catch {}
                }, 5000);
                break;
            case 'node':
            case 'javascript':
            case 'js':
                // Write code to temp file and execute
                const tempJsFile = path.join(os.tmpdir(), `skill_${Date.now()}_${Math.random().toString(36).substring(7)}.js`);
                fs.writeFileSync(tempJsFile, code);
                command = `node "${tempJsFile}"`;
                // Clean up temp file after execution
                setTimeout(() => {
                    try { fs.unlinkSync(tempJsFile); } catch {}
                }, 5000);
                break;
            case 'typescript':
            case 'ts':
                // Write code to temp file and execute with tsx
                const tempTsFile = path.join(os.tmpdir(), `skill_${Date.now()}_${Math.random().toString(36).substring(7)}.ts`);
                fs.writeFileSync(tempTsFile, code);
                command = `npx tsx "${tempTsFile}"`;
                // Clean up temp file after execution
                setTimeout(() => {
                    try { fs.unlinkSync(tempTsFile); } catch {}
                }, 5000);
                break;
            default:
                return {
                    success: false,
                    output: '',
                    error: `Unsupported script language: ${language}. Supported: bash, python, node, typescript`
                };
        }
        
        console.log(`🚀 Executing ${language} script in ${workingDir}`);
        const { stdout, stderr } = await execAsync(command, {
            cwd: workingDir,
            maxBuffer: 10 * 1024 * 1024 // 10MB buffer
        });
        
        return {
            success: true,
            output: stdout || stderr || 'Script executed successfully',
            error: stderr || undefined
        };
    } catch (error: any) {
        return {
            success: false,
            output: '',
            error: error.message || 'Script execution failed'
        };
    }
}

/**
 * Create a tool from a skill
 */
function createToolFromSkill(skillInfo: SkillInfo) {
    return tool({
        description: skillInfo.description + (skillInfo.hasExecutableScripts ? ' (Contains executable scripts)' : ''),
        inputSchema: z.object({
            task: z.string().describe('The task or request that should be handled by this skill'),
            context: z.string().optional().describe('Additional context or parameters for the skill'),
            executeScript: z.boolean().optional().describe('If true and skill contains executable scripts, execute them directly. Default: false'),
            scriptIndex: z.number().optional().describe('Index of the script to execute (0-based). If not specified, executes all executable scripts.'),
        }),
        execute: async ({ task, context, executeScript: shouldExecute, scriptIndex }) => {
            try {
                console.log(`🎯 Executing skill: ${skillInfo.name}`);
                console.log(`   Task: ${task}`);
                if (context) console.log(`   Context: ${context}`);
                
                // If skill has executable scripts and user wants to execute them
                if (shouldExecute && skillInfo.hasExecutableScripts && skillInfo.codeBlocks.length > 0) {
                    const executableLanguages = ['bash', 'sh', 'shell', 'python', 'python3', 'node', 'javascript', 'js', 'typescript', 'ts'];
                    const scriptsToExecute = scriptIndex !== undefined
                        ? [skillInfo.codeBlocks[scriptIndex]].filter(b => b && (executableLanguages.includes(b.language) || b.language === ''))
                        : skillInfo.codeBlocks.filter(b => executableLanguages.includes(b.language) || b.language === '');
                    
                    const executionResults = [];
                    
                    for (const script of scriptsToExecute) {
                        console.log(`   Executing script ${script.index} (${script.language})`);
                        const result = await executeScript(script.language, script.code);
                        executionResults.push({
                            index: script.index,
                            language: script.language,
                            code: script.code,
                            ...result
                        });
                    }
                    
                    return {
                        success: true,
                        skillName: skillInfo.name,
                        skillDescription: skillInfo.description,
                        instructions: skillInfo.content,
                        task: task,
                        context: context || '',
                        scriptsExecuted: executionResults.length,
                        executionResults: executionResults,
                        message: `Skill ${skillInfo.name} executed. ${executionResults.length} script(s) executed.`
                    };
                }
                
                // Return the skill content and instructions (default behavior)
                return {
                    success: true,
                    skillName: skillInfo.name,
                    skillDescription: skillInfo.description,
                    instructions: skillInfo.content,
                    codeBlocks: skillInfo.codeBlocks.map(b => ({
                        language: b.language,
                        code: b.code,
                        index: b.index,
                        executable: ['bash', 'sh', 'shell', 'python', 'python3', 'node', 'javascript', 'js', 'typescript', 'ts', ''].includes(b.language)
                    })),
                    hasExecutableScripts: skillInfo.hasExecutableScripts,
                    task: task,
                    context: context || '',
                    message: `Skill ${skillInfo.name} loaded. ${skillInfo.hasExecutableScripts ? `Contains ${skillInfo.codeBlocks.filter(b => ['bash', 'sh', 'shell', 'python', 'python3', 'node', 'javascript', 'js', 'typescript', 'ts', ''].includes(b.language)).length} executable script(s). Set executeScript=true to run them.` : 'Follow the instructions in the skill content to complete the task.'}`
                };
            } catch (error: any) {
                return {
                    success: false,
                    error: error.message || 'Failed to execute skill',
                    skillName: skillInfo.name
                };
            }
        },
    });
}

// Load skills at startup
const loadedSkills = loadSkills();
const skillTools: Record<string, any> = {};

// Create tools from loaded skills
for (const [skillKey, skillInfo] of Object.entries(loadedSkills)) {
    skillTools[skillKey] = createToolFromSkill(skillInfo);
}

// --- Skills Integration (End) ---

// Combined tools (base + MCP tools + skills)
function getTools() {
    return {
        ...baseTools,
        ...mcpTools,
        ...skillTools,
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
          5. ALWAYS provide clear feedback after completing tasks - tell the user what was done and the result.
          
          Available tools: ${toolNames.join(', ')}
          
          Skills (skill_*): These are specialized Cursor skills that provide detailed instructions for specific tasks.
          When a skill is called, it returns instructions that you should follow to complete the task.
          Examples: skill_create-rule (create Cursor rules), skill_create-skill (create new skills), etc.
          
          For git operations:
          - Use git-mcp tools (git-mcp_*) to check status, add files, commit, and push
          - If git-mcp tools are available, use them directly without asking
          - For "提交代码" or "commit code", you should: check status → add files → commit → push
          
          For file operations: use readFile, writeFile, deleteFile tools.
          
          For specialized tasks (creating rules, skills, etc.):
          - Use skill_* tools when available
          - When a skill is called, read the returned instructions carefully
          - Follow the skill's instructions step by step to complete the task
          
          Current working directory: ${PROJECT_ROOT}
          User home directory: ${os.homedir()}
          
          WORKFLOW FOR ACTION REQUESTS:
          1. Identify which tools are needed
          2. Call the tools immediately to gather information or perform actions
          3. Continue calling tools until the task is complete
          4. ALWAYS provide a clear summary after completion:
             - What was done
             - Success or failure status
             - Any important results or outputs
             - Next steps if applicable
          
          FEEDBACK REQUIREMENTS:
          - After executing tools, ALWAYS provide a summary message to the user
          - Include the results from tool executions in your response
          - If a task completes successfully, clearly state "任务已完成" or "Task completed"
          - If there are errors, explain what went wrong
          - Never end a response without telling the user what happened
          
          REMEMBER: You are an executor, not a consultant. When asked to do something, DO IT using tools, then TELL THE USER what happened.`,
            messages: messages,
            tools: availableTools,
            toolChoice: requiresAction ? 'required' : 'auto', // Force tool usage for action requests
            stopWhen: stepCountIs(10), // Increased from 5 to allow more tool calls
        });

        if (toolCalls && toolCalls.length > 0) {
            console.log('\n🛠️ Tools used:', toolCalls.map(c => c.toolName).join(', '));
            console.log(`📊 Total tool calls: ${toolCalls.length}`);
        }

        // Ensure agent provides feedback if tools were used
        const hasToolCalls = toolCalls && toolCalls.length > 0;
        const hasTextResponse = text && text.trim().length > 0;
        
        // If tools were used but no text response, add a default feedback message
        let finalText = text;
        if (hasToolCalls && !hasTextResponse) {
            finalText = `✅ 已完成 ${toolCalls.length} 个工具调用。任务执行完成。`;
            console.log('⚠️  No text response after tool calls, adding default feedback');
        }

        console.log(`\n✨ Agent response generated${hasToolCalls ? ' (with tool calls)' : ''}`);

        ctx.body = {
            messages: response.messages,
            text: finalText || text,
            toolCalls: toolCalls || [],
            hasToolCalls: hasToolCalls
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
