import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Archive,
  ArrowLeft,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  FileBarChart,
  FileImage,
  FileText,
  GitBranch,
  Home,
  LayoutDashboard,
  Loader2,
  LogOut,
  Maximize2,
  MessageSquare,
  Moon,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Search,
  Send,
  Sparkles,
  Sun,
  UploadCloud,
  X,
} from 'lucide-react';
import { DotField } from './components/DotField';
import { SpotlightCard } from './components/SpotlightCard';
import { supabase } from './lib/supabaseClient';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const ACCEPTED_TYPES = '.pdf,.docx,.png,.jpg,.jpeg,.xlsx,.csv,.txt,.pptx';

const getWorkspaceTheme = (isLightMode) => ({
  bg: isLightMode ? '#FAFAF8' : '#0B0B0C',
  surface: isLightMode ? '#FFFFFF' : '#131416',
  card: isLightMode ? '#F5F5F4' : '#1A1B1E',
  accent: isLightMode ? '#D97706' : '#F59E0B',
  text: isLightMode ? '#18181B' : '#FAFAF9',
  secondary: isLightMode ? '#71717A' : '#A1A1AA',
  border: isLightMode ? 'rgba(24,24,27,0.10)' : 'rgba(250,250,249,0.10)',
  softBorder: isLightMode ? 'rgba(24,24,27,0.07)' : 'rgba(250,250,249,0.07)',
  dotFrom: isLightMode ? 'rgba(217,119,6,0.12)' : 'rgba(245,158,11,0.14)',
  dotTo: isLightMode ? 'rgba(217,119,6,0.05)' : 'rgba(245,158,11,0.05)',
});

const getFileIcon = (name) => {
  const ext = name.split('.').pop()?.toLowerCase();
  if (['png', 'jpg', 'jpeg'].includes(ext)) return FileImage;
  if (['xlsx', 'csv'].includes(ext)) return FileBarChart;
  return FileText;
};

const shortName = (name) => (name.length > 24 ? `${name.slice(0, 16)}...${name.slice(-5)}` : name);

const IconButton = ({ title, children, className = '', style, ...props }) => (
  <button
    title={title}
    className={`group relative flex h-9 w-9 items-center justify-center rounded-xl border transition-colors ${className}`}
    style={style}
    {...props}
  >
    {children}
  </button>
);

const SidebarItem = ({ icon: Icon, label, collapsed, active, theme }) => (
  <button
    title={collapsed ? label : undefined}
    className="group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-colors"
    style={{
      color: active ? theme.text : theme.secondary,
      background: active ? `${theme.accent}16` : 'transparent',
    }}
  >
    <Icon size={18} style={{ color: active ? theme.accent : theme.secondary }} />
    {!collapsed && <span className="truncate">{label}</span>}
  </button>
);

const WorkspaceGraph = ({ graphData, theme, compact = false }) => {
  if (!graphData?.nodes?.length) {
    return (
      <div className="flex h-full flex-col items-center justify-center text-center" style={{ color: theme.secondary }}>
        <Network size={34} className="mb-3 opacity-60" />
        <p className="text-sm">Ask a question to generate a knowledge graph.</p>
      </div>
    );
  }

  const size = compact ? 240 : 300;
  const center = size / 2;
  const radius = compact ? 76 : 100;

  return (
    <svg className="h-full min-h-[240px] w-full" viewBox={`0 0 ${size} ${size}`}>
      {graphData.edges?.map((edge, i) => {
        const sourceIndex = graphData.nodes.findIndex((node) => node.node_id === edge.source);
        const targetIndex = graphData.nodes.findIndex((node) => node.node_id === edge.target);
        const getPos = (idx) => {
          const safeIndex = idx === -1 ? 0 : idx;
          const angle = (safeIndex / graphData.nodes.length) * 2 * Math.PI - Math.PI / 2;
          return { x: center + radius * Math.cos(angle), y: center + radius * Math.sin(angle) };
        };
        const source = getPos(sourceIndex);
        const target = getPos(targetIndex);
        return (
          <motion.line
            key={`${edge.source}-${edge.target}-${i}`}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            stroke={theme.accent}
            strokeOpacity="0.34"
            strokeWidth="1.2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.7, delay: i * 0.04 }}
          />
        );
      })}
      {graphData.nodes.map((node, i) => {
        const angle = (i / graphData.nodes.length) * 2 * Math.PI - Math.PI / 2;
        const x = center + radius * Math.cos(angle);
        const y = center + radius * Math.sin(angle);
        const label = node.label || node.node_id || 'Node';
        return (
          <motion.g key={node.node_id || i} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.08 * i }}>
            <circle cx={x} cy={y} r={i === 0 ? 13 : 9} fill={theme.card} stroke={theme.accent} strokeWidth="1.4" />
            <text x={x} y={y + 22} fill={theme.secondary} fontSize="7" textAnchor="middle" fontWeight="600">
              {label.length > 18 ? `${label.slice(0, 18)}...` : label}
            </text>
          </motion.g>
        );
      })}
    </svg>
  );
};

export const DashboardWorkspace = ({ setView, session, isLightMode, setIsLightMode }) => {
  const theme = getWorkspaceTheme(isLightMode);
  const [activeTab, setActiveTab] = useState('source');
  const [activeCitation, setActiveCitation] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [docs, setDocs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState(() => localStorage.getItem('documind-left-collapsed') === 'true');
  const [rightCollapsed, setRightCollapsed] = useState(() => {
    const stored = localStorage.getItem('documind-right-collapsed');
    if (stored) return stored === 'true';
    return window.innerWidth < 1100;
  });
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('documind-left-collapsed', String(leftCollapsed));
  }, [leftCollapsed]);

  useEffect(() => {
    localStorage.setItem('documind-right-collapsed', String(rightCollapsed));
  }, [rightCollapsed]);

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = '0px';
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
  }, [input]);

  const activeCitationData = useMemo(() => {
    if (!activeCitation) return null;
    for (const msg of messages) {
      const found = msg.citations?.find((citation) => citation.id === activeCitation);
      if (found) return found;
    }
    return null;
  }, [activeCitation, messages]);

  const citations = useMemo(() => messages.flatMap((msg) => msg.citations || []), [messages]);

  const uploadFiles = useCallback(async (fileList) => {
    const files = Array.from(fileList || []);
    if (!files.length) return;

    for (const file of files) {
      const Icon = getFileIcon(file.name);
      const docId = `${Date.now()}-${file.name}`;
      const newDoc = {
        id: docId,
        name: file.name,
        status: 'Uploading',
        progress: 18,
        iconName: Icon.name,
      };
      setDocs((prev) => [newDoc, ...prev]);

      const progressTimer = setInterval(() => {
        setDocs((prev) => prev.map((doc) => (doc.id === docId && doc.status === 'Uploading' ? { ...doc, progress: Math.min((doc.progress || 18) + 12, 86) } : doc)));
      }, 350);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch(`${API_BASE}/api/ingest`, {
          method: 'POST',
          body: formData,
        });
        clearInterval(progressTimer);
        if (res.ok) {
          setDocs((prev) => prev.map((doc) => (doc.id === docId ? { ...doc, status: 'Parsed', progress: 100 } : doc)));
        } else {
          const err = await res.json().catch(() => ({}));
          setDocs((prev) => prev.map((doc) => (doc.id === docId ? { ...doc, status: err.detail ? `Error: ${err.detail}` : 'Error', progress: 100 } : doc)));
        }
      } catch (err) {
        clearInterval(progressTimer);
        console.error(err);
        setDocs((prev) => prev.map((doc) => (doc.id === docId ? { ...doc, status: 'Error', progress: 100 } : doc)));
      }
    }
  }, []);

  const handleSend = async (overrideText) => {
    const query = (overrideText || input).trim();
    if (!query || isProcessing) return;

    const messageIndex = messages.length + 1;
    setMessages((prev) => [...prev, { sender: 'user', text: query }, { sender: 'ai', text: '', status: 'Connecting to document intelligence...', citations: null }]);
    setInput('');
    setIsProcessing(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, agentic: false }),
      });

      if (!res.ok) throw new Error('Network response was not ok');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let currentText = '';

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (!value) continue;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        let currentEvent = null;

        for (const line of lines) {
          if (line.startsWith('event: ')) currentEvent = line.substring(7).trim();
          if (!line.startsWith('data: ')) continue;

          const dataStr = line.substring(6).trim();
          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);
            if (currentEvent === 'status') {
              setMessages((prev) => {
                const next = [...prev];
                next[messageIndex] = { ...next[messageIndex], status: data.message };
                return next;
              });
            } else if (currentEvent === 'answer') {
              currentText += data.text;
              setMessages((prev) => {
                const next = [...prev];
                next[messageIndex] = { ...next[messageIndex], text: currentText, status: null };
                return next;
              });
            } else if (currentEvent === 'error') {
              setMessages((prev) => {
                const next = [...prev];
                next[messageIndex] = { ...next[messageIndex], text: `Error: ${data.message}`, status: null };
                return next;
              });
            } else if (currentEvent === 'done') {
              if (data.metadata?.sources) {
                const mappedCitations = data.metadata.sources.map((source) => ({
                  id: source.marker,
                  label: `[${source.marker}]`,
                  type: 'text',
                  title: source.title || source.source || 'Extracted Source',
                  snippet: source.snippet,
                }));
                setMessages((prev) => {
                  const next = [...prev];
                  next[messageIndex] = { ...next[messageIndex], citations: mappedCitations };
                  return next;
                });
              }
              if (data.metadata?.contributing_subgraph) setGraphData(data.metadata.contributing_subgraph);
            }
          } catch {
            // Ignore partial stream chunks.
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => {
        const next = [...prev];
        next[messageIndex] = { ...next[messageIndex], text: 'Failed to connect to backend.', status: null };
        return next;
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleWorkspaceDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    uploadFiles(event.dataTransfer.files);
  };

  const quickActions = [
    ['Analyze a Research Paper', '/analyze research paper'],
    ['Summarize a Contract', '/summary contract'],
    ['Extract Key Insights', '/extract key insights'],
    ['Build a Knowledge Graph', '/graph uploaded documents'],
  ];

  const commands = ['/summary', '/extract', '/graph', '/analyze'];
  const recentDocs = docs.slice(0, 5);

  const renderDocIcon = (name) => {
    const Icon = getFileIcon(name);
    return <Icon size={16} />;
  };

  return (
    <div
      className="relative flex h-screen w-full overflow-hidden font-sans"
      style={{ background: theme.bg, color: theme.text }}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setIsDragging(false);
      }}
      onDrop={handleWorkspaceDrop}
    >
      <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.12]">
        <DotField
          glowColor={theme.bg}
          gradientFrom={theme.dotFrom}
          gradientTo={theme.dotTo}
          dotRadius={1.35}
          dotSpacing={22}
          cursorRadius={300}
          cursorForce={0.03}
          bulgeStrength={18}
        />
      </div>

      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-3 z-50 flex items-center justify-center rounded-[28px] border border-dashed backdrop-blur-sm"
            style={{ borderColor: theme.accent, background: `${theme.bg}CC`, color: theme.text }}
          >
            <div className="text-center">
              <UploadCloud size={34} className="mx-auto mb-3" style={{ color: theme.accent }} />
              <p className="text-lg font-semibold">Drop files to add them to this workspace</p>
              <p className="mt-1 text-sm" style={{ color: theme.secondary }}>PDF, DOCX, images, spreadsheets, CSV, TXT, and PPTX</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.aside
        animate={{ width: leftCollapsed ? 56 : 280 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 flex shrink-0 flex-col border-r"
        style={{ background: theme.surface, borderColor: theme.border }}
      >
        <div className="flex h-16 items-center justify-between px-3">
          <button onClick={() => setView('landing')} className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl" style={{ background: `${theme.accent}16`, color: theme.accent }}>
              <BrainCircuit size={19} />
            </span>
            {!leftCollapsed && <span className="truncate text-sm font-semibold">DocuMind</span>}
          </button>
          {!leftCollapsed && (
            <IconButton
              title="Collapse sidebar"
              onClick={() => setLeftCollapsed(true)}
              style={{ borderColor: theme.softBorder, color: theme.secondary }}
            >
              <PanelLeftClose size={17} />
            </IconButton>
          )}
        </div>

        {leftCollapsed && (
          <button
            title="Expand sidebar"
            onClick={() => setLeftCollapsed(false)}
            className="mx-auto mb-3 flex h-9 w-9 items-center justify-center rounded-xl border"
            style={{ borderColor: theme.softBorder, color: theme.secondary }}
          >
            <PanelLeftOpen size={17} />
          </button>
        )}

        <div className="space-y-1 px-2">
          <SidebarItem icon={LayoutDashboard} label="Workspace" collapsed={leftCollapsed} active theme={theme} />
          <SidebarItem icon={Search} label="Recent Documents" collapsed={leftCollapsed} theme={theme} />
          <SidebarItem icon={Archive} label="Upload History" collapsed={leftCollapsed} theme={theme} />
          <SidebarItem icon={BookOpen} label="Knowledge Library" collapsed={leftCollapsed} theme={theme} />
        </div>

        {!leftCollapsed && (
          <>
            <div className="mt-6 px-4">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: theme.secondary }}>Recent</p>
                <button onClick={() => fileInputRef.current?.click()} className="text-xs font-semibold" style={{ color: theme.accent }}>Upload</button>
              </div>
              <div className="space-y-2">
                {recentDocs.length === 0 ? (
                  <div className="rounded-2xl border p-4 text-sm" style={{ borderColor: theme.softBorder, color: theme.secondary, background: theme.card }}>
                    No documents yet.
                  </div>
                ) : recentDocs.map((doc) => (
                  <div key={doc.id} className="rounded-2xl border p-3" style={{ borderColor: theme.softBorder, background: theme.card }}>
                    <div className="flex items-center gap-3">
                      <span style={{ color: theme.accent }}>{renderDocIcon(doc.name)}</span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium">{doc.name}</p>
                        <div className="mt-1 flex items-center gap-2 text-xs" style={{ color: theme.secondary }}>
                          {doc.status === 'Parsed' ? <CheckCircle2 size={12} style={{ color: theme.accent }} /> : doc.status === 'Error' || doc.status?.startsWith('Error') ? <X size={12} /> : <Loader2 size={12} className="animate-spin" />}
                          {doc.status}
                        </div>
                      </div>
                    </div>
                    {doc.status === 'Uploading' && (
                      <div className="mt-3 h-1 overflow-hidden rounded-full" style={{ background: `${theme.accent}18` }}>
                        <div className="h-full rounded-full transition-all" style={{ width: `${doc.progress || 0}%`, background: theme.accent }} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-auto border-t p-4" style={{ borderColor: theme.softBorder }}>
              <div className="flex items-center gap-3 rounded-2xl border p-3" style={{ borderColor: theme.softBorder, background: theme.card }}>
                <div className="flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold" style={{ background: `${theme.accent}18`, color: theme.accent }}>
                  {(session?.user?.email || 'A').charAt(0).toUpperCase()}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold">Analyst User</p>
                  <p className="truncate text-xs" style={{ color: theme.secondary }}>{session?.user?.email || 'analyst@enterprise.com'}</p>
                </div>
                <button
                  title="Log out"
                  onClick={async () => {
                    if (supabase) await supabase.auth.signOut();
                    setView('landing');
                  }}
                  style={{ color: theme.secondary }}
                >
                  <LogOut size={16} />
                </button>
              </div>
            </div>
          </>
        )}
      </motion.aside>

      <main className="relative z-10 flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b px-5" style={{ borderColor: theme.border, background: `${theme.bg}E6` }}>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-base font-semibold">Document Intelligence Workspace</h1>
              <span className="rounded-full px-2.5 py-1 text-xs font-semibold" style={{ background: `${theme.accent}14`, color: theme.accent }}>Reasoning</span>
            </div>
            <p className="mt-0.5 text-xs" style={{ color: theme.secondary }}>Upload, ask, cite, and map evidence in one flow.</p>
          </div>
          <div className="flex items-center gap-2">
            <IconButton
              title="Toggle theme"
              onClick={() => setIsLightMode(!isLightMode)}
              style={{ borderColor: theme.softBorder, color: theme.secondary, background: theme.surface }}
            >
              {isLightMode ? <Moon size={17} /> : <Sun size={17} />}
            </IconButton>
            <IconButton
              title="Home"
              onClick={() => setView('landing')}
              style={{ borderColor: theme.softBorder, color: theme.secondary, background: theme.surface }}
            >
              <Home size={17} />
            </IconButton>
          </div>
        </header>

        <section className="flex min-h-0 flex-1 flex-col">
          <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col overflow-hidden px-5">
            <div className="flex-1 overflow-y-auto py-8">
              {messages.length === 0 ? (
                <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="flex min-h-full items-center justify-center">
                  <div className="w-full max-w-2xl text-center">
                    <div className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl" style={{ background: `${theme.accent}14`, color: theme.accent }}>
                      <Sparkles size={26} />
                    </div>
                    <h2 className="text-3xl font-semibold tracking-tight">What would you like to understand?</h2>
                    <p className="mx-auto mt-3 max-w-xl text-base leading-7" style={{ color: theme.secondary }}>
                      Upload documents or start with a command. DocuMind will connect sources, citations, and reasoning as the conversation develops.
                    </p>
                    <div className="mt-8 grid gap-3 sm:grid-cols-2">
                      {quickActions.map(([label, command]) => (
                        <button
                          key={label}
                          onClick={() => setInput(command)}
                          className="rounded-2xl border p-4 text-left text-sm font-semibold transition-transform hover:-translate-y-0.5"
                          style={{ borderColor: theme.softBorder, background: theme.surface, color: theme.text }}
                        >
                          {label}
                          <p className="mt-1 text-xs font-medium" style={{ color: theme.secondary }}>{command}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ) : (
                <div className="space-y-8">
                  {messages.map((msg, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-[82%] ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}>
                        <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em]" style={{ color: theme.secondary }}>
                          {msg.sender === 'user' ? 'You' : 'DocuMind'}
                        </div>
                        <div
                          className="rounded-3xl border px-5 py-4 text-[15px] leading-7 shadow-sm"
                          style={{
                            background: msg.sender === 'user' ? `${theme.accent}18` : theme.surface,
                            borderColor: msg.sender === 'user' ? `${theme.accent}40` : theme.softBorder,
                            color: theme.text,
                          }}
                        >
                          {msg.status ? (
                            <div className="flex items-center gap-2" style={{ color: theme.secondary }}>
                              <Loader2 size={16} className="animate-spin" style={{ color: theme.accent }} />
                              {msg.status}
                            </div>
                          ) : msg.text}
                          {msg.citations?.length > 0 && (
                            <div className="mt-4 flex flex-wrap gap-2 border-t pt-3" style={{ borderColor: theme.softBorder }}>
                              {msg.citations.map((citation) => (
                                <button
                                  key={citation.id}
                                  onClick={() => {
                                    setActiveCitation(citation.id);
                                    setActiveTab('source');
                                    setRightCollapsed(false);
                                  }}
                                  className="rounded-full border px-3 py-1 text-xs font-semibold"
                                  style={{ borderColor: `${theme.accent}35`, color: theme.accent, background: `${theme.accent}10` }}
                                >
                                  {citation.label || citation.id}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            <div className="pb-5">
              {docs.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">
                  {docs.slice(0, 6).map((doc) => (
                    <span key={doc.id} className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium" style={{ borderColor: theme.softBorder, background: theme.surface, color: theme.text }}>
                      <span style={{ color: theme.accent }}>{renderDocIcon(doc.name)}</span>
                      {shortName(doc.name)}
                      {doc.status === 'Uploading' && <Loader2 size={12} className="animate-spin" style={{ color: theme.accent }} />}
                      <button onClick={() => setDocs((prev) => prev.filter((docItem) => docItem.id !== doc.id))} style={{ color: theme.secondary }}>
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
              )}

              <SpotlightCard className="rounded-[26px] border shadow-xl shadow-black/5" style={{ borderColor: theme.border, background: theme.surface }}>
                <div className="p-3">
                  <div className="mb-2 flex flex-wrap gap-2 px-2">
                    {commands.map((command) => (
                      <button
                        key={command}
                        onClick={() => setInput(command)}
                        className="rounded-full border px-3 py-1 text-xs font-semibold transition-colors"
                        style={{ borderColor: theme.softBorder, color: theme.secondary }}
                      >
                        {command}
                      </button>
                    ))}
                  </div>
                  <div className="flex items-end gap-2">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="mb-1 flex h-10 shrink-0 items-center gap-2 rounded-2xl border px-3 text-sm font-semibold transition-colors"
                      style={{ borderColor: `${theme.accent}38`, color: theme.accent, background: `${theme.accent}10` }}
                    >
                      <Plus size={17} /> Upload
                    </button>
                    <textarea
                      ref={textareaRef}
                      value={input}
                      rows={1}
                      onChange={(event) => setInput(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.shiftKey) {
                          event.preventDefault();
                          handleSend();
                        }
                      }}
                      placeholder="Ask anything about your documents... Use /commands or @mentions"
                      className="max-h-[180px] min-h-11 flex-1 resize-none bg-transparent px-2 py-3 text-[15px] outline-none"
                      style={{ color: theme.text }}
                    />
                    <button
                      onClick={() => handleSend()}
                      disabled={!input.trim() || isProcessing}
                      className="mb-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl transition-opacity disabled:opacity-45"
                      style={{ background: theme.accent, color: isLightMode ? '#FFFFFF' : '#0B0B0C' }}
                    >
                      {isProcessing ? <Loader2 size={18} className="animate-spin" /> : <Send size={17} />}
                    </button>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept={ACCEPTED_TYPES}
                    className="hidden"
                    onChange={(event) => {
                      uploadFiles(event.target.files);
                      event.target.value = '';
                    }}
                  />
                </div>
              </SpotlightCard>
            </div>
          </div>
        </section>
      </main>

      <motion.aside
        animate={{ width: rightCollapsed ? 56 : 320 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 flex shrink-0 flex-col border-l"
        style={{ background: theme.surface, borderColor: theme.border }}
      >
        <div className="flex h-16 items-center justify-between px-3">
          {!rightCollapsed && <p className="text-sm font-semibold">Context</p>}
          <IconButton
            title={rightCollapsed ? 'Expand context panel' : 'Collapse context panel'}
            onClick={() => setRightCollapsed(!rightCollapsed)}
            style={{ borderColor: theme.softBorder, color: theme.secondary }}
          >
            {rightCollapsed ? <PanelRightOpen size={17} /> : <PanelRightClose size={17} />}
          </IconButton>
        </div>

        {rightCollapsed ? (
          <div className="flex flex-col items-center gap-2 px-2">
            {[
              ['source', FileText, 'Source Viewer'],
              ['graph', GitBranch, 'Knowledge Graph'],
              ['citations', MessageSquare, 'Citations'],
            ].map(([tab, Icon, label]) => (
              <button
                key={tab}
                title={label}
                onClick={() => {
                  setActiveTab(tab);
                  setRightCollapsed(false);
                }}
                className="flex h-10 w-10 items-center justify-center rounded-xl border"
                style={{ borderColor: activeTab === tab ? `${theme.accent}50` : theme.softBorder, color: activeTab === tab ? theme.accent : theme.secondary }}
              >
                <Icon size={17} />
              </button>
            ))}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-1 px-3 pb-3">
              {[
                ['source', FileText, 'Source'],
                ['graph', GitBranch, 'Graph'],
                ['citations', MessageSquare, 'Citations'],
              ].map(([tab, Icon, label]) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className="rounded-xl px-2 py-2 text-xs font-semibold transition-colors"
                  style={{ background: activeTab === tab ? `${theme.accent}16` : 'transparent', color: activeTab === tab ? theme.accent : theme.secondary }}
                >
                  <Icon size={15} className="mx-auto mb-1" />
                  {label}
                </button>
              ))}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">
              <AnimatePresence mode="wait">
                {activeTab === 'source' && (
                  <motion.div key="source" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} className="h-full">
                    {activeCitationData ? (
                      <div className="rounded-3xl border p-4" style={{ borderColor: theme.softBorder, background: theme.card }}>
                        <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em]" style={{ color: theme.accent }}>
                          <FileText size={14} /> {activeCitationData.title || 'Extracted Source'}
                        </div>
                        <p className="border-l-2 pl-3 text-sm leading-7" style={{ borderColor: `${theme.accent}55`, color: theme.secondary }}>
                          {activeCitationData.snippet || 'No snippet provided for this source.'}
                        </p>
                      </div>
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center text-center" style={{ color: theme.secondary }}>
                        <FileText size={34} className="mb-3 opacity-60" />
                        <p className="text-sm">Select a citation to inspect source evidence.</p>
                      </div>
                    )}
                  </motion.div>
                )}

                {activeTab === 'graph' && (
                  <motion.div key="graph" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} className="h-full rounded-3xl border p-3" style={{ borderColor: theme.softBorder, background: theme.card }}>
                    <WorkspaceGraph graphData={graphData} theme={theme} />
                  </motion.div>
                )}

                {activeTab === 'citations' && (
                  <motion.div key="citations" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} className="space-y-2">
                    {citations.length === 0 ? (
                      <div className="flex min-h-[360px] flex-col items-center justify-center text-center" style={{ color: theme.secondary }}>
                        <MessageSquare size={34} className="mb-3 opacity-60" />
                        <p className="text-sm">Citations will appear after grounded answers.</p>
                      </div>
                    ) : citations.map((citation) => (
                      <button
                        key={citation.id}
                        onClick={() => {
                          setActiveCitation(citation.id);
                          setActiveTab('source');
                        }}
                        className="w-full rounded-2xl border p-3 text-left"
                        style={{ borderColor: theme.softBorder, background: theme.card }}
                      >
                        <p className="text-sm font-semibold">{citation.title || citation.id}</p>
                        <p className="mt-1 line-clamp-2 text-xs leading-5" style={{ color: theme.secondary }}>{citation.snippet || 'Source excerpt available.'}</p>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </>
        )}
      </motion.aside>
    </div>
  );
};
