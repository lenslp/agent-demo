import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles } from 'lucide-react';

// Update Message interface to support AI SDK format
interface Message {
    role: string; // 'user' | 'assistant' | 'tool' | 'system'
    content: string | Array<any>;
}

function App() {
    const [messages, setMessages] = useState<Message[]>([
        { role: 'assistant', content: 'Hello! I am your AI assistant. How can I help you today?' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage: Message = { role: 'user', content: input };
        const newMessages = [...messages, userMessage];

        setMessages(newMessages);
        setInput('');
        setIsLoading(true);

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3001';
            const response = await fetch(`${apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ messages: newMessages }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // data.messages contains the new messages from the agent (including tool calls/results)
            // We just need to add them to our conversation history
            if (data.messages && Array.isArray(data.messages)) {
                setMessages(prev => [...prev, ...data.messages]);
            } else if (data.text) {
                // Fallback if structure is different
                setMessages(prev => [...prev, { role: 'assistant', content: data.text }]);
            }
        } catch (error) {
            console.error('Error calling agent:', error);
            setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I encountered an error connecting to the agent. Make sure the backend server is running." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const renderMessageContent = (content: any) => {
        if (typeof content === 'string') return content;
        if (Array.isArray(content)) {
            return content.map((part, i) => {
                if (part.type === 'text') return <span key={i}>{part.text}</span>;
                if (part.type === 'tool-call') return <span key={i} className="italic text-xs block opacity-70">🛠️ Calling tool: {part.toolName}...</span>;
                return null;
            });
        }
        return JSON.stringify(content);
    };

    return (
        <div className="flex h-screen bg-background text-foreground font-sans">
            {/* Sidebar - mimicking Cursor/VS Code sidebar */}
            <div className="w-16 border-r border-border flex flex-col items-center py-4 gap-4 bg-secondary/30">
                <div className="p-2 rounded-md bg-primary/10 text-primary">
                    <Bot size={24} />
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full h-full relative">
                {/* Header */}
                <header className="border-b border-border p-4 flex items-center gap-2 bg-background/80 backdrop-blur-sm z-10 sticky top-0">
                    <Sparkles className="text-primary w-5 h-5" />
                    <h1 className="font-semibold text-lg">Agent Chat</h1>
                </header>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-6">
                    {messages.map((message, index) => {
                        // Skip system messages or hidden messages if desired, but here we show mostly user/assistant
                        if (message.role === 'system') return null;

                        const isUser = message.role === 'user';
                        const isTool = message.role === 'tool';

                        return (
                            <div
                                key={index}
                                className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
                            >
                                {!isTool && (
                                    <div className={`
                                        w-8 h-8 rounded-full flex items-center justify-center shrink-0
                                        ${isUser ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground'}
                                    `}>
                                        {isUser ? <User size={16} /> : <Bot size={16} />}
                                    </div>
                                )}

                                <div className={`
                                    px-4 py-2.5 rounded-2xl max-w-[80%] text-sm leading-relaxed
                                    ${isUser
                                        ? 'bg-primary text-primary-foreground rounded-tr-none'
                                        : isTool
                                            ? 'bg-muted text-muted-foreground w-full border border-dashed text-xs font-mono'
                                            : 'bg-secondary/50 border border-border rounded-tl-none'}
                                `}>
                                    {isTool ? (
                                        <div className="flex items-center gap-2">
                                            <span>🔧 Tool Result</span>
                                            {/* We could hide the result or show it condensed */}
                                        </div>
                                    ) : renderMessageContent(message.content)}
                                </div>
                            </div>
                        );
                    })}

                    {isLoading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-secondary text-secondary-foreground flex items-center justify-center shrink-0">
                                <Bot size={16} />
                            </div>
                            <div className="bg-secondary/50 border border-border px-4 py-3 rounded-2xl rounded-tl-none flex items-center gap-1">
                                <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 border-t border-border bg-background">
                    <form onSubmit={handleSubmit} className="relative">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask anything..."
                            className="w-full bg-secondary/30 border border-border rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all font-light"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-primary text-primary-foreground rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            <Send size={16} />
                        </button>
                    </form>
                    <div className="text-center mt-2 text-xs text-muted-foreground">
                        AI can make mistakes. Please verify important information.
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
