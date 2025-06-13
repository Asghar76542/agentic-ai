import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

function App() {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentView, setCurrentView] = useState('chat');
    const [responseData, setResponseData] = useState(null);
    const [isOnline, setIsOnline] = useState(false);
    const [status, setStatus] = useState('Agents ready');
    const [expandedReasoning, setExpandedReasoning] = useState(new Set());
    const [darkMode, setDarkMode] = useState(true);
    const messagesEndRef = useRef(null);

    const normalizeAnswer = (answer) => {
        return answer
            .trim()
            .toLowerCase()
            .replace(/\s+/g, ' ')
            .replace(/[.,!?]/g, '')
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const toggleReasoning = (messageIndex) => {
        setExpandedReasoning(prev => {
            const newSet = new Set(prev);
            if (newSet.has(messageIndex)) {
                newSet.delete(messageIndex);
            } else {
                newSet.add(messageIndex);
            }
            return newSet;
        });
    };

    const checkHealth = async () => {
        try {
            await axios.get(`${BACKEND_URL}/health`);
            setIsOnline(true);
            console.log('System is online');
        } catch {
            setIsOnline(false);
            console.log('System is offline');
        }
    };

    const fetchScreenshot = async () => {
        try {
            const timestamp = new Date().getTime();
            const res = await axios.get(`${BACKEND_URL}/screenshots/updated_screen.png?timestamp=${timestamp}`, {
                responseType: 'blob'
            });
            console.log('Screenshot fetched successfully');
            const imageUrl = URL.createObjectURL(res.data);
            setResponseData((prev) => {
                if (prev?.screenshot && prev.screenshot !== 'placeholder.png') {
                    URL.revokeObjectURL(prev.screenshot);
                }
                return {
                    ...prev,
                    screenshot: imageUrl,
                    screenshotTimestamp: new Date().getTime()
                };
            });
        } catch (err) {
            console.error('Error fetching screenshot:', err);
            setResponseData((prev) => ({
                ...prev,
                screenshot: 'placeholder.png',
                screenshotTimestamp: new Date().getTime()
            }));
        }
    };

    const fetchLatestAnswer = useCallback(async () => {
        try {
            const res = await axios.get(`${BACKEND_URL}/latest_answer`);
            const data = res.data;

            updateData(data);
            if (!data.answer || data.answer.trim() === '') {
                return;
            }
            const normalizedNewAnswer = normalizeAnswer(data.answer);
            const answerExists = messages.some(
                (msg) => normalizeAnswer(msg.content) === normalizedNewAnswer
            );
            if (!answerExists) {
                setMessages((prev) => [
                    ...prev,
                    {
                        type: 'agent',
                        content: data.answer,
                        reasoning: data.reasoning,
                        agentName: data.agent_name,
                        status: data.status,
                        uid: data.uid,
                    },
                ]);
                setStatus(data.status);
                scrollToBottom();
            } else {
                console.log('Duplicate answer detected, skipping:', data.answer);
            }
        } catch (error) {
            console.error('Error fetching latest answer:', error);
        }
    }, [messages]);

    useEffect(() => {
        const intervalId = setInterval(() => {
            checkHealth();
            fetchLatestAnswer();
            fetchScreenshot();
        }, 3000);
        return () => clearInterval(intervalId);
    }, [fetchLatestAnswer]);

    const updateData = (data) => {
        setResponseData((prev) => ({
            ...prev,
            blocks: data.blocks || prev.blocks || null,
            done: data.done,
            answer: data.answer,
            agent_name: data.agent_name,
            status: data.status,
            uid: data.uid,
        }));
    };

    const handleStop = async (e) => {
        e.preventDefault();
        checkHealth();
        setIsLoading(false);
        setError(null);
        try {
            await axios.get(`${BACKEND_URL}/stop`);
            setStatus("Requesting stop...");
        } catch (err) {
            console.error('Error stopping the agent:', err);
        }
    }

    const handleOpenDashboard = async () => {
        try {
            await axios.get(`${BACKEND_URL}/memory_dashboard`);
            window.open(`${BACKEND_URL}/dashboard/memory_dashboard.html`, '_blank');
        } catch (err) {
            console.error('Error opening dashboard:', err);
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        checkHealth();
        if (!query.trim()) {
            console.log('Empty query');
            return;
        }
        setMessages((prev) => [...prev, { type: 'user', content: query }]);
        setIsLoading(true);
        setError(null);

        try {
            console.log('Sending query:', query);
            setQuery('waiting for response...');
            const res = await axios.post(`${BACKEND_URL}/query`, {
                query,
                tts_enabled: false
            });
            setQuery('Enter your query...');
            console.log('Response:', res.data);
            const data = res.data;
            updateData(data);
        } catch (err) {
            console.error('Error:', err);
            setError('Failed to process query.');
            setMessages((prev) => [
                ...prev,
                { type: 'error', content: 'Error: Unable to get a response.' },
            ]);
        } finally {
            console.log('Query completed');
            setIsLoading(false);
            setQuery('');
        }
    };

    const handleGetScreenshot = async () => {
        try {
            setCurrentView('screenshot');
        } catch (err) {
            setError('Browser not in use');
        }
    };

    return (
        <div className={`app ${darkMode ? 'dark-mode' : ''}`}>
            <header className="header">
                <div className="header-left">
                    <h1>AgenticSeek</h1>
                    <div className="status-indicator">
                        <div className={`status-dot ${isOnline ? '' : 'disconnected'}`}></div>
                        <span>{isOnline ? status : 'System Offline'}</span>
                    </div>
                </div>
                <div className="header-right">
                    <button
                        className="theme-toggle"
                        onClick={() => setDarkMode(!darkMode)}
                        title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
                    >
                        {darkMode ? '☀️' : '🌙'}
                    </button>
                    <button
                        className="dashboard-button"
                        onClick={handleOpenDashboard}
                        title="Open Memory Dashboard"
                    >
                        📊
                    </button>
                </div>
            </header>
            <main className="main">
                {currentView === 'chat' ? (
                    <div className="chat-view">
                        <div className="chat-section">
                            <h2>Chat Interface</h2>
                            <div className="messages">
                                {messages.length === 0 ? (
                                    <p className="placeholder">
                                        🚀 Ready to help! Ask me anything or give me a task to complete.
                                    </p>
                                ) : (
                                    messages.map((msg, index) => (
                                        <div
                                            key={index}
                                            className={`message ${
                                                msg.type === 'user'
                                                    ? 'user-message'
                                                    : msg.type === 'agent'
                                                    ? `agent-message ${msg.agentName ? msg.agentName.toLowerCase().replace(/\s+/g, '-') + '-agent' : ''}`
                                                    : 'error-message'
                                            }`}
                                        >
                                            {msg.type === 'agent' && (
                                                <div className="message-header">
                                                    {msg.agentName && (
                                                        <span className="agent-name">{msg.agentName}</span>
                                                    )}
                                                    {msg.reasoning && (
                                                        <button 
                                                            className="reasoning-toggle"
                                                            onClick={() => toggleReasoning(index)}
                                                            title={expandedReasoning.has(index) ? "Hide reasoning" : "Show reasoning"}
                                                        >
                                                            {expandedReasoning.has(index) ? '▼' : '▶'} Reasoning
                                                        </button>
                                                    )}
                                                </div>
                                            )}
                                            <div className="message-content">
                                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                                            </div>
                                            {msg.type === 'agent' && msg.reasoning && expandedReasoning.has(index) && (
                                                <div className="reasoning-content">
                                                    <ReactMarkdown>{msg.reasoning}</ReactMarkdown>
                                                </div>
                                            )}
                                        </div>
                                    ))
                                )}
                                <div ref={messagesEndRef} />
                            </div>
                            {isLoading && (
                                <div className="loading-animation">
                                    <div className="typing-indicator">
                                        <span></span>
                                        <span></span>
                                        <span></span>
                                    </div>
                                    Processing your request...
                                </div>
                            )}
                            {isOnline && !isLoading && (
                                <div className="loading-animation">{status}</div>
                            )}
                            {!isOnline && (
                                <p className="loading-animation">
                                    ⚠️ System offline. Deploy backend first.
                                </p>
                            )}
                            <form onSubmit={handleSubmit} className="input-form">
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Type your query or task here..."
                                    disabled={isLoading}
                                />
                                <button type="submit" disabled={isLoading || !isOnline}>
                                    Send
                                </button>
                                <button type="button" onClick={handleStop} disabled={!isLoading}>
                                    Stop
                                </button>
                            </form>
                        </div>
                    </div>
                ) : (
                    <div className="app-sections">
                        <div className="task-section">
                            <h2>Task Details</h2>
                            <div className="task-details">
                                {responseData?.answer ? (
                                    <div>
                                        <strong>Current Task:</strong>
                                        <br />
                                        {responseData.answer}
                                        <br /><br />
                                        <strong>Agent:</strong> {responseData.agent_name || 'Unknown'}
                                        <br />
                                        <strong>Status:</strong> {responseData.status || 'Unknown'}
                                        <br />
                                        <strong>UID:</strong> {responseData.uid || 'N/A'}
                                    </div>
                                ) : (
                                    <div>No active task. Start a conversation to see task details here.</div>
                                )}
                            </div>
                        </div>

                        <div className="computer-section">
                            <h2>Computer View</h2>
                            <div className="view-selector">
                                <button
                                    className={currentView === 'blocks' ? 'active' : ''}
                                    onClick={() => setCurrentView('blocks')}
                                >
                                    📝 Editor View
                                </button>
                                <button
                                    className={currentView === 'screenshot' ? 'active' : ''}
                                    onClick={responseData?.screenshot ? () => setCurrentView('screenshot') : handleGetScreenshot}
                                >
                                    🖥️ Browser View
                                </button>
                            </div>
                            <div className="content">
                                {error && <p className="error">{error}</p>}
                                {currentView === 'blocks' ? (
                                    <div className="blocks">
                                        {responseData && responseData.blocks && Object.values(responseData.blocks).length > 0 ? (
                                            Object.values(responseData.blocks).map((block, index) => (
                                                <div key={index} className="block">
                                                    <p className="block-tool">Tool: {block.tool_type}</p>
                                                    <pre>{block.block}</pre>
                                                    <p className="block-feedback">Feedback: {block.feedback}</p>
                                                    {block.success ? (
                                                        <p className="block-success">Success</p>
                                                    ) : (
                                                        <p className="block-failure">Failure</p>
                                                    )}
                                                </div>
                                            ))
                                        ) : (
                                            <div className="block">
                                                <p className="block-tool">Tool: No tool in use</p>
                                                <pre>No file opened or tool activity yet.</pre>
                                                <p className="block-feedback">Start a task to see editor activity here.</p>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="screenshot-container">
                                        <img
                                            src={responseData?.screenshot || 'placeholder.png'}
                                            alt="Browser Screenshot"
                                            onError={(e) => {
                                                e.target.src = 'placeholder.png';
                                                console.error('Failed to load screenshot');
                                            }}
                                            key={responseData?.screenshotTimestamp || 'default'}
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
                
                <div className="section-tabs">
                    <button
                        className={currentView === 'chat' ? 'active' : ''}
                        onClick={() => setCurrentView('chat')}
                    >
                        💬 Chat
                    </button>
                    <button
                        className={currentView === 'blocks' || currentView === 'screenshot' ? 'active' : ''}
                        onClick={() => setCurrentView('blocks')}
                    >
                        🔧 Workspace
                    </button>
                </div>
            </main>
        </div>
    );
}

export default App;