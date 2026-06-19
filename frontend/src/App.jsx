import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence, useInView, useMotionValue, useTransform } from 'framer-motion';
import {
  UploadCloud, FileText, FileImage, FileBarChart,
  Send, BrainCircuit, Network, LayoutDashboard,
  ChevronRight, Lock, User, CheckCircle2, Loader2,
  Sparkles, Zap, Shield, ArrowRight, Play, Star,
  Layers, Search, MessageSquare, Database, Globe, Cpu,
  X, ArrowLeft, Mail, BookOpen, Scale, FileCheck, Copy
} from 'lucide-react';

// --- MOCK DATA ---
const mockDocs = [
  { id: 1, name: 'Q3_Financial_Report.pdf', status: 'Parsed', icon: <FileBarChart size={18} /> },
  { id: 2, name: 'System_Architecture_v2.docx', status: 'Extracting Tables...', icon: <FileText size={18} /> },
  { id: 3, name: 'Server_Topography.png', status: 'Parsed', icon: <FileImage size={18} /> },
];

const mockChat = [
  { sender: 'user', text: 'What was our primary revenue driver in Q3, and how does it relate to the new server architecture?' },
  {
    sender: 'ai',
    text: 'Based on the uploaded documents, the primary revenue driver in Q3 was Enterprise Cloud Solutions, which saw a 34% increase. This aligns directly with the deployment of the distributed server topography. You can see the revenue breakdown in ',
    citations: [
      { id: 'c1', label: '[Table 3, Page 12]' },
      { id: 'c2', label: '[Architecture Diagram]' }
    ]
  }
];

// --- COMPONENTS ---

const GlassCard = ({ children, className = "" }) => (
  <div className={`bg-white/10 backdrop-blur-lg border border-white/20 shadow-2xl rounded-2xl ${className}`}>
    {children}
  </div>
);

// --- ANIMATED DOT GRID BACKGROUND ---
const DotGrid = () => {
  const canvasRef = useRef(null);
  const mouseRef = useRef({ x: -1000, y: -1000 });
  const animFrameRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let cols, rows;
    const spacing = 40;
    const baseRadius = 1.2;
    const hoverRadius = 3.5;
    const hoverRange = 120;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = document.querySelector('.landing-scroll')?.scrollHeight || window.innerHeight * 4;
      cols = Math.ceil(canvas.width / spacing);
      rows = Math.ceil(canvas.height / spacing);
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const rect = canvas.getBoundingClientRect();

      let mx = -1000;
      let my = -1000;
      if (mouseRef.current.x !== -1000) {
        mx = mouseRef.current.x - rect.left;
        my = mouseRef.current.y - rect.top;
      }

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * spacing + spacing / 2;
          const y = r * spacing + spacing / 2;
          const dx = x - mx;
          const dy = y - my;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const t = Math.max(0, 1 - dist / hoverRange);
          const radius = baseRadius + (hoverRadius - baseRadius) * t;
          const alpha = 0.08 + 0.35 * t;
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(136, 189, 242, ${alpha})`;
          ctx.fill();
        }
      }
      animFrameRef.current = requestAnimationFrame(draw);
    };

    resize();
    draw();
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  const handleMouseMove = useCallback((e) => {
    mouseRef.current = { x: e.clientX, y: e.clientY };
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full pointer-events-none z-0"
    />
  );
};

// --- ANIMATED COUNTER ---
const AnimCounter = ({ target, suffix = "", duration = 2 }) => {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-50px" });

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const step = target / (duration * 60);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 1000 / 60);
    return () => clearInterval(timer);
  }, [inView, target, duration]);

  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
};

// --- FLOATING ORB ---
const FloatingOrb = ({ className, delay = 0 }) => (
  <motion.div
    className={`absolute rounded-full pointer-events-none blur-3xl ${className}`}
    animate={{
      y: [0, -30, 0, 20, 0],
      x: [0, 15, -10, 5, 0],
      scale: [1, 1.1, 0.95, 1.05, 1],
    }}
    transition={{
      duration: 12,
      repeat: Infinity,
      delay,
      ease: "easeInOut",
    }}
  />
);

// --- PIPELINE NODE ---
const PipelineNode = ({ icon: Icon, label, color, delay, index }) => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-30px" });
  return (
    <div ref={ref} className="flex flex-col items-center gap-3 relative">
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={inView ? { scale: 1, opacity: 1 } : {}}
        transition={{ delay: delay, type: "spring", stiffness: 300, damping: 20 }}
        className={`relative w-20 h-20 md:w-24 md:h-24 rounded-2xl border-2 flex items-center justify-center ${color} backdrop-blur-sm`}
      >
        {/* Pulse ring */}
        <motion.div
          className={`absolute inset-0 rounded-2xl ${color}`}
          animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0, 0.5] }}
          transition={{ duration: 2.5, repeat: Infinity, delay: delay + 0.5 }}
        />
        <Icon size={32} className="relative z-10" />
      </motion.div>
      <motion.span
        initial={{ opacity: 0, y: 10 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ delay: delay + 0.2 }}
        className="text-xs md:text-sm font-semibold text-storm-100/80 text-center"
      >
        {label}
      </motion.span>
    </div>
  );
};

// --- PIPELINE CONNECTOR ---
const PipelineConnector = ({ delay }) => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-30px" });
  return (
    <div ref={ref} className="flex items-center justify-center flex-1 max-w-[80px] md:max-w-[120px] relative h-20 md:h-24">
      <motion.div
        initial={{ scaleX: 0 }}
        animate={inView ? { scaleX: 1 } : {}}
        transition={{ delay, duration: 0.5, ease: "easeOut" }}
        className="h-[2px] w-full bg-gradient-to-r from-storm-300/60 to-storm-500/40 origin-left"
      />
      {/* Traveling dot */}
      <motion.div
        initial={{ left: "0%", opacity: 0 }}
        animate={inView ? { left: ["0%", "100%"], opacity: [0, 1, 1, 0] } : {}}
        transition={{ delay: delay + 0.3, duration: 1.2, repeat: Infinity, repeatDelay: 2 }}
        className="absolute w-2 h-2 bg-storm-300 rounded-full shadow-[0_0_10px_rgba(136,189,242,0.8)]"
        style={{ top: "calc(50% - 4px)" }}
      />
    </div>
  );
};

// 1. LANDING PAGE
const LandingPage = ({ setView }) => {
  const [scrollY, setScrollY] = useState(0);
  const [showDemo, setShowDemo] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [footerModal, setFooterModal] = useState(null); // 'privacy' | 'terms' | 'docs' | 'contact' | null
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // ESC key to close demo modal
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') setShowDemo(false); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  const features = [
    {
      icon: FileImage,
      title: "Visual Parsing",
      description: "Extract semantic meaning from charts, diagrams, and embedded images using state-of-the-art multi-modal models.",
      hex: "#88BDF2",
      borderColor: "hover:border-storm-300/50",
      glowColor: "from-storm-300/10",
      iconColor: "text-storm-300",
    },
    {
      icon: Network,
      title: "Knowledge Graph",
      description: "Automatically map relationships between paragraphs, tables, footnotes, and figures into a traversable graph.",
      hex: "#34d399",
      borderColor: "hover:border-emerald-400/50",
      glowColor: "from-emerald-400/10",
      iconColor: "text-emerald-400",
    },
    {
      icon: BrainCircuit,
      title: "Agentic QA",
      description: "Ask multi-hop questions across documents and get fully cited, explainable answers with reasoning chains.",
      hex: "#c084fc",
      borderColor: "hover:border-purple-400/50",
      glowColor: "from-purple-400/10",
      iconColor: "text-purple-400",
    },
    {
      icon: Layers,
      title: "Table Extraction",
      description: "Precisely extract and reconstruct complex tables from PDFs, scans, and images—preserving structure and hierarchy.",
      hex: "#fbbf24",
      borderColor: "hover:border-amber-400/50",
      glowColor: "from-amber-400/10",
      iconColor: "text-amber-400",
    },
    {
      icon: Shield,
      title: "Enterprise Security",
      description: "SOC-2 compliant, end-to-end encrypted processing. Your documents never leave your tenant boundary.",
      hex: "#22d3ee",
      borderColor: "hover:border-cyan-400/50",
      glowColor: "from-cyan-400/10",
      iconColor: "text-cyan-400",
    },
    {
      icon: Globe,
      title: "Multi-Language",
      description: "Process documents in 50+ languages with cross-lingual understanding—query in English, parse in any language.",
      hex: "#fb7185",
      borderColor: "hover:border-rose-400/50",
      glowColor: "from-rose-400/10",
      iconColor: "text-rose-400",
    },
  ];

  const steps = [
    {
      num: "01",
      title: "Upload Documents",
      desc: "Drag-and-drop PDFs, DOCX, images, or spreadsheets. Our engine automatically detects document type and structure.",
      icon: UploadCloud,
      accentHex: "#88BDF2",
    },
    {
      num: "02",
      title: "AI Parses & Connects",
      desc: "Multi-modal models extract text, tables, charts, and images—then build a semantic knowledge graph linking every element.",
      icon: Cpu,
      accentHex: "#34d399",
    },
    {
      num: "03",
      title: "Ask Anything",
      desc: "Query across all your documents in natural language. Get cited, explainable answers with source traceability.",
      icon: MessageSquare,
      accentHex: "#c084fc",
    },
  ];

  const stats = [
    { value: 98, suffix: "%", label: "Extraction Accuracy" },
    { value: 50, suffix: "+", label: "Languages Supported" },
    { value: 3, suffix: "s", label: "Avg. Parse Time" },
    { value: 10, suffix: "M+", label: "Documents Processed" },
  ];

  const staggerContainer = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const staggerItem = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
  };

  return (
    <div className="landing-scroll relative bg-[#0f1923] overflow-x-hidden font-sans selection:bg-storm-300 selection:text-storm-900">

      {/* Dot Grid Background */}
      <DotGrid />

      {/* Floating Ambient Orbs */}
      <FloatingOrb className="w-[500px] h-[500px] bg-storm-300/8 top-[-100px] left-[-150px]" delay={0} />
      <FloatingOrb className="w-[400px] h-[400px] bg-purple-500/6 top-[30%] right-[-100px]" delay={3} />
      <FloatingOrb className="w-[350px] h-[350px] bg-emerald-400/6 bottom-[20%] left-[10%]" delay={6} />

      {/* ====== STICKY NAVBAR ====== */}
      <motion.nav
        initial={{ y: -80 }}
        animate={{ y: 0 }}
        transition={{ delay: 0.2, type: "spring", stiffness: 120 }}
        className="fixed top-0 left-0 right-0 z-50 px-6 py-4"
      >
        <div
          className={`max-w-7xl mx-auto flex items-center justify-between rounded-2xl px-6 py-3 transition-all duration-500 ${scrollY > 50
            ? 'bg-[#0f1923]/80 backdrop-blur-xl border border-white/10 shadow-2xl shadow-black/20'
            : 'bg-transparent'
            }`}
        >
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-storm-300 to-storm-500 flex items-center justify-center shadow-lg shadow-storm-300/20">
              <BrainCircuit size={20} className="text-white" />
            </div>
            <span className="text-white font-bold text-lg tracking-tight hidden sm:inline">DocuMind</span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setView('login')}
              className="text-storm-100/70 hover:text-white text-sm font-medium px-4 py-2 rounded-xl hover:bg-white/5 transition-all"
            >
              Sign In
            </button>
            <button
              onClick={() => setShowOnboarding(true)}
              className="bg-storm-300 hover:bg-storm-100 text-[#0f1923] font-bold text-sm px-5 py-2.5 rounded-xl transition-all shadow-lg shadow-storm-300/20 hover:shadow-storm-300/40"
            >
              Get Started
            </button>
          </div>
        </div>
      </motion.nav>

      {/* ====== HERO SECTION ====== */}
      <section className="relative z-10 min-h-screen flex flex-col items-center justify-center px-6 pt-28 pb-20">
        {/* Top Ambient Glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-storm-300/10 blur-[150px] rounded-full pointer-events-none" />

        <div className="max-w-5xl mx-auto flex flex-col items-center text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="inline-flex items-center gap-2.5 px-5 py-2.5 rounded-full border border-storm-300/20 bg-storm-300/5 backdrop-blur-md mb-8"
          >
            <Sparkles size={16} className="text-storm-300" />
            <span className="text-storm-100/90 text-xs font-bold tracking-[0.2em] uppercase">
              Powered by Multi-Modal AI
            </span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black text-white mb-8 tracking-tighter leading-[0.95]"
          >
            Decode the{' '}
            <br className="hidden sm:block" />
            <span className="relative inline-block">
              <span
                className="animate-gradient-shift"
                style={{
                  background: 'linear-gradient(90deg, #88BDF2, #a8d8ff, #c084fc)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  backgroundSize: '200% 200%',
                }}
              >
                Unstructured.
              </span>
              {/* Underline glow */}
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: 1, duration: 0.8 }}
                className="absolute -bottom-2 left-0 right-0 h-[3px] origin-left rounded-full"
                style={{ background: 'linear-gradient(90deg, #88BDF2, #c084fc, transparent)' }}
              />
            </span>
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="text-lg md:text-xl text-storm-100/60 mb-12 leading-relaxed max-w-2xl font-medium"
          >
            Upload any document—PDFs, scans, spreadsheets, images—and let our AI
            build a{' '}
            <span className="text-storm-300 font-semibold">living knowledge graph</span>{' '}
            you can query in natural language.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="flex flex-col sm:flex-row items-center gap-4"
          >
            <button
              onClick={() => setView('login')}
              className="group relative inline-flex items-center justify-center gap-3 text-[#0f1923] hover:text-[#0f1923] font-extrabold text-lg px-10 py-5 rounded-2xl transition-all duration-300 shadow-[0_0_50px_rgba(136,189,242,0.25)] hover:shadow-[0_0_80px_rgba(136,189,242,0.45)] overflow-hidden"
              style={{ background: 'linear-gradient(90deg, #88BDF2, #6ba3e0)' }}
            >
              <span className="relative z-10 flex items-center gap-2">
                Start Analyzing
                <ArrowRight size={22} className="group-hover:translate-x-1.5 transition-transform duration-300" />
              </span>
              <div className="absolute inset-0 -translate-x-full group-hover:animate-[shimmer_1.5s_infinite] bg-gradient-to-r from-transparent via-white/30 to-transparent z-0" />
            </button>

            <button
              onClick={() => setShowDemo(true)}
              className="group inline-flex items-center gap-2.5 text-storm-100/70 hover:text-white font-semibold text-lg px-6 py-5 rounded-2xl hover:bg-white/5 transition-all border border-transparent hover:border-white/10"
            >
              <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center group-hover:bg-white/20 transition-colors">
                <Play size={16} className="text-white ml-0.5" />
              </div>
              Watch Demo
            </button>
          </motion.div>

          {/* Trust Line */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            className="mt-12 flex items-center gap-2 text-storm-100/30 text-sm"
          >
            <Shield size={14} />
            <span>SOC-2 Compliant · End-to-End Encrypted · No data retention</span>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 rounded-full border-2 border-storm-100/20 flex justify-center pt-2"
          >
            <div className="w-1 h-2 rounded-full bg-storm-300/60" />
          </motion.div>
        </motion.div>
      </section>

      {/* ====== PIPELINE VISUALIZATION ====== */}
      <section className="relative z-10 py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            className="text-center mb-16"
          >
            <span className="text-storm-300 text-sm font-bold tracking-[0.2em] uppercase mb-4 block">The Pipeline</span>
            <h2 className="text-3xl md:text-5xl font-black text-white tracking-tight">
              From Raw Documents to{' '}
              <span
                style={{
                  background: 'linear-gradient(90deg, #34d399, #88BDF2)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Actionable Intelligence
              </span>
            </h2>
          </motion.div>

          {/* Pipeline Flow */}
          <div className="flex items-center justify-center flex-wrap md:flex-nowrap gap-2 md:gap-0">
            <PipelineNode icon={UploadCloud} label="Upload" color="bg-storm-300/10 border-storm-300/30 text-storm-300" delay={0} index={0} />
            <PipelineConnector delay={0.2} />
            <PipelineNode icon={Search} label="Parse" color="bg-emerald-400/10 border-emerald-400/30 text-emerald-400" delay={0.4} index={1} />
            <PipelineConnector delay={0.6} />
            <PipelineNode icon={Database} label="Graph Build" color="bg-purple-400/10 border-purple-400/30 text-purple-400" delay={0.8} index={2} />
            <PipelineConnector delay={1.0} />
            <PipelineNode icon={BrainCircuit} label="Reason" color="bg-amber-400/10 border-amber-400/30 text-amber-400" delay={1.2} index={3} />
            <PipelineConnector delay={1.4} />
            <PipelineNode icon={MessageSquare} label="Answer" color="bg-cyan-400/10 border-cyan-400/30 text-cyan-400" delay={1.6} index={4} />
          </div>
        </div>
      </section>

      {/* ====== FEATURES BENTO GRID ====== */}
      <section className="relative z-10 py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            className="text-center mb-16"
          >
            <span className="text-storm-300 text-sm font-bold tracking-[0.2em] uppercase mb-4 block">Capabilities</span>
            <h2 className="text-3xl md:text-5xl font-black text-white tracking-tight">
              Everything You Need to{' '}
              <span
                style={{
                  background: 'linear-gradient(90deg, #88BDF2, #c084fc)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Understand Documents
              </span>
            </h2>
          </motion.div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
          >
            {features.map((feat, idx) => (
              <motion.div
                key={idx}
                variants={staggerItem}
                whileHover={{ y: -6, scale: 1.01 }}
                className={`group relative p-7 rounded-3xl bg-white/[0.03] border border-white/[0.06] ${feat.borderColor} hover:bg-white/[0.06] transition-all duration-500 backdrop-blur-sm overflow-hidden cursor-default text-left`}
              >
                {/* Hover glow */}
                <div className={`absolute inset-0 bg-gradient-to-br ${feat.glowColor} to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                {/* Top-right corner glow */}
                <div
                  className="absolute -top-12 -right-12 w-32 h-32 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                  style={{ backgroundColor: `${feat.hex}15` }}
                />

                <div
                  className="w-12 h-12 rounded-2xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform duration-300"
                  style={{
                    backgroundColor: `${feat.hex}18`,
                    border: `1px solid ${feat.hex}30`,
                  }}
                >
                  <feat.icon style={{ color: feat.hex }} size={24} />
                </div>
                <h3 className="text-white font-bold text-xl mb-3 relative z-10">{feat.title}</h3>
                <p className="text-storm-100/50 text-sm leading-relaxed relative z-10">{feat.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ====== HOW IT WORKS ====== */}
      <section className="relative z-10 py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            className="text-center mb-20"
          >
            <span className="text-storm-300 text-sm font-bold tracking-[0.2em] uppercase mb-4 block">How It Works</span>
            <h2 className="text-3xl md:text-5xl font-black text-white tracking-tight">
              Three Steps to{' '}
              <span
                style={{
                  background: 'linear-gradient(90deg, #88BDF2, #34d399)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Total Clarity
              </span>
            </h2>
          </motion.div>

          <div className="relative">
            {/* Vertical line */}
            <div className="absolute left-6 md:left-1/2 md:-translate-x-px top-0 bottom-0 w-[2px] bg-gradient-to-b from-storm-300/30 via-emerald-400/30 to-purple-400/30 hidden md:block" />

            <div className="space-y-16 md:space-y-24">
              {steps.map((step, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: idx % 2 === 0 ? -40 : 40 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: "-80px" }}
                  transition={{ duration: 0.6, delay: 0.1 }}
                  className={`flex items-center gap-8 md:gap-16 ${idx % 2 === 1 ? 'md:flex-row-reverse' : ''}`}
                >
                  <div className="flex-1">
                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-3xl p-8 hover:bg-white/[0.06] transition-all duration-300">
                      <div className="text-sm font-black tracking-[0.15em] uppercase mb-3" style={{ color: step.accentHex }}>{step.num}</div>
                      <h3 className="text-white text-2xl font-bold mb-3">{step.title}</h3>
                      <p className="text-storm-100/50 leading-relaxed">{step.desc}</p>
                    </div>
                  </div>

                  {/* Center icon node */}
                  <div className="hidden md:flex flex-shrink-0 w-14 h-14 rounded-2xl bg-[#0f1923] border-2 border-storm-300/30 items-center justify-center z-10 shadow-xl shadow-storm-300/10">
                    <step.icon size={24} style={{ color: step.accentHex }} />
                  </div>

                  <div className="flex-1 hidden md:block" />
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ====== STATS BAR ====== */}
      <section className="relative z-10 py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="bg-white/[0.03] border border-white/[0.06] rounded-3xl p-10 md:p-14">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-4">
              {stats.map((stat, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: idx * 0.1 }}
                  className="text-center"
                >
                  <div className="text-4xl md:text-5xl font-black text-white mb-2">
                    <AnimCounter target={typeof stat.value === 'number' ? stat.value : 0} suffix={stat.suffix} />
                  </div>
                  <div className="text-storm-100/40 text-sm font-medium">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ====== BOTTOM CTA ====== */}
      <section className="relative z-10 py-32 px-6">
        <div className="max-w-4xl mx-auto text-center relative">
          {/* Background glow */}
          <div className="absolute inset-0 -top-20 -bottom-20 bg-gradient-to-r from-storm-300/10 via-purple-400/5 to-emerald-400/10 blur-[100px] rounded-full pointer-events-none" />

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-6xl font-black text-white tracking-tight mb-6">
              Ready to Decode{' '}
              <span
                style={{
                  background: 'linear-gradient(90deg, #88BDF2, #c084fc)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Your Data?
              </span>
            </h2>
            <p className="text-storm-100/50 text-lg mb-10 max-w-xl mx-auto">
              Join thousands of enterprises transforming how they understand unstructured documents.
            </p>
            <button
              onClick={() => setView('login')}
              className="group relative inline-flex items-center justify-center gap-3 text-[#0f1923] font-extrabold text-xl px-14 py-6 rounded-2xl transition-all duration-300 shadow-[0_0_60px_rgba(136,189,242,0.3)] hover:shadow-[0_0_100px_rgba(136,189,242,0.5)] overflow-hidden"
              style={{ background: 'linear-gradient(90deg, #88BDF2, #6ba3e0)' }}
            >
              <span className="relative z-10 flex items-center gap-2">
                Initialize Workspace
                <ChevronRight size={24} className="group-hover:translate-x-2 transition-transform duration-300" />
              </span>
              <div className="absolute inset-0 -translate-x-full group-hover:animate-[shimmer_1.5s_infinite] bg-gradient-to-r from-transparent via-white/30 to-transparent z-0" />
            </button>
          </motion.div>
        </div>
      </section>

      {/* ====== FOOTER ====== */}
      <footer className="relative z-10 border-t border-white/[0.06] py-10 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-storm-300 to-storm-500 flex items-center justify-center">
              <BrainCircuit size={16} className="text-white" />
            </div>
            <span className="text-white font-bold">DocuMind</span>
            <span className="text-storm-100/30 text-sm">© 2026</span>
          </div>
          <div className="flex items-center gap-6 text-storm-100/40 text-sm">
            <button onClick={() => setFooterModal('privacy')} className="hover:text-storm-100 transition-colors cursor-pointer">Privacy</button>
            <button onClick={() => setFooterModal('terms')} className="hover:text-storm-100 transition-colors cursor-pointer">Terms</button>
            <button onClick={() => setFooterModal('docs')} className="hover:text-storm-100 transition-colors cursor-pointer">Documentation</button>
            <button onClick={() => setFooterModal('contact')} className="hover:text-storm-100 transition-colors cursor-pointer">Contact</button>
          </div>
        </div>
      </footer>

      {/* ====== ONBOARDING MODAL (Get Started) ====== */}
      <AnimatePresence>
        {showOnboarding && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4"
            onClick={() => setShowOnboarding(false)}
          >
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" />
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="relative z-10 w-full max-w-2xl max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="bg-[#0f1923] border border-white/10 rounded-3xl p-8 md:p-10 shadow-2xl">
                {/* Close */}
                <button
                  onClick={() => setShowOnboarding(false)}
                  className="absolute top-6 right-6 w-10 h-10 rounded-xl bg-white/5 hover:bg-white/10 flex items-center justify-center text-storm-100/60 hover:text-white transition-all"
                >
                  <X size={20} />
                </button>

                <div className="text-center mb-8">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-storm-300 to-storm-500 flex items-center justify-center mx-auto mb-5 shadow-lg shadow-storm-300/20">
                    <BrainCircuit size={32} className="text-white" />
                  </div>
                  <h2 className="text-3xl font-black text-white mb-3">Welcome to DocuMind</h2>
                  <p className="text-storm-100/60">Intelligent Unstructured Document Understanding powered by Multi-Modal AI</p>
                </div>

                {/* What it does */}
                <div className="space-y-4 mb-8">
                  <h3 className="text-white font-bold text-lg flex items-center gap-2">
                    <Sparkles size={18} style={{ color: '#88BDF2' }} /> What does DocuMind do?
                  </h3>
                  <p className="text-storm-100/60 text-sm leading-relaxed">
                    DocuMind uses advanced multi-modal AI to parse, understand, and connect information from your unstructured documents — including PDFs, Word files, scanned images, charts, and tables. It builds a semantic knowledge graph from your documents so you can ask questions in plain English and get cited, explainable answers.
                  </p>
                </div>

                {/* How to upload */}
                <div className="space-y-4 mb-8">
                  <h3 className="text-white font-bold text-lg flex items-center gap-2">
                    <UploadCloud size={18} style={{ color: '#34d399' }} /> How to Upload Documents
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-start gap-3 bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm font-black" style={{ backgroundColor: '#88BDF218', color: '#88BDF2' }}>1</div>
                      <div>
                        <p className="text-white font-semibold text-sm">Sign in to your workspace</p>
                        <p className="text-storm-100/50 text-xs mt-1">Create an account or sign in to access the dashboard.</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm font-black" style={{ backgroundColor: '#34d39918', color: '#34d399' }}>2</div>
                      <div>
                        <p className="text-white font-semibold text-sm">Drag & drop your files</p>
                        <p className="text-storm-100/50 text-xs mt-1">Upload PDFs, DOCX, PNG, or XLSX files into the workspace panel on the left side.</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm font-black" style={{ backgroundColor: '#c084fc18', color: '#c084fc' }}>3</div>
                      <div>
                        <p className="text-white font-semibold text-sm">Ask questions & get answers</p>
                        <p className="text-storm-100/50 text-xs mt-1">Use the chat interface to query across all your documents. Get cited answers with source traceability.</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Supported formats */}
                <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4 mb-8">
                  <h4 className="text-white font-semibold text-sm mb-3 flex items-center gap-2">
                    <FileCheck size={16} style={{ color: '#fbbf24' }} /> Supported Formats
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {['PDF', 'DOCX', 'PNG', 'JPG', 'XLSX', 'CSV', 'TXT', 'PPTX'].map(fmt => (
                      <span key={fmt} className="px-3 py-1 rounded-lg bg-white/[0.05] border border-white/[0.08] text-storm-100/70 text-xs font-mono">{fmt}</span>
                    ))}
                  </div>
                </div>

                <button
                  onClick={() => { setShowOnboarding(false); setView('login'); }}
                  className="w-full text-[#0f1923] hover:text-[#0f1923] font-bold py-4 rounded-xl transition-all shadow-lg shadow-storm-300/20 hover:shadow-storm-300/40 text-lg flex items-center justify-center gap-2 hover:opacity-90"
                  style={{ background: 'linear-gradient(90deg, #88BDF2, #6A89A7)' }}
                >
                  Continue to Sign In <ArrowRight size={20} />
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ====== FOOTER MODALS ====== */}
      <AnimatePresence>
        {footerModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4"
            onClick={() => setFooterModal(null)}
          >
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" />
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="relative z-10 w-full max-w-xl max-h-[85vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="bg-[#0f1923] border border-white/10 rounded-3xl p-8 shadow-2xl">
                {/* Close */}
                <button
                  onClick={() => setFooterModal(null)}
                  className="absolute top-6 right-6 w-10 h-10 rounded-xl bg-white/5 hover:bg-white/10 flex items-center justify-center text-storm-100/60 hover:text-white transition-all"
                >
                  <X size={20} />
                </button>

                {/* Privacy */}
                {footerModal === 'privacy' && (
                  <div>
                    <div className="flex items-center gap-3 mb-6">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: '#88BDF218' }}>
                        <Shield size={20} style={{ color: '#88BDF2' }} />
                      </div>
                      <h2 className="text-2xl font-bold text-white">Privacy Policy</h2>
                    </div>
                    <div className="space-y-4 text-storm-100/60 text-sm leading-relaxed">
                      <p><strong className="text-white">Data Protection:</strong> All documents uploaded to DocuMind are processed with end-to-end encryption. We do not store your documents after processing unless explicitly requested.</p>
                      <p><strong className="text-white">No Third-Party Sharing:</strong> Your data is never shared with third parties, advertisers, or external services. All processing happens within your secure tenant boundary.</p>
                      <p><strong className="text-white">SOC-2 Compliance:</strong> DocuMind is fully SOC-2 Type II compliant, ensuring enterprise-grade security controls, audit logging, and access management.</p>
                      <p><strong className="text-white">Data Retention:</strong> We follow a zero-retention policy by default. Uploaded documents are purged from our servers immediately after processing. Users may opt-in to persistent storage for workspace continuity.</p>
                      <p><strong className="text-white">User Rights:</strong> You have the right to access, correct, or delete your personal data at any time. Contact us to exercise these rights.</p>
                    </div>
                  </div>
                )}

                {/* Terms */}
                {footerModal === 'terms' && (
                  <div>
                    <div className="flex items-center gap-3 mb-6">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: '#c084fc18' }}>
                        <Scale size={20} style={{ color: '#c084fc' }} />
                      </div>
                      <h2 className="text-2xl font-bold text-white">Terms of Service</h2>
                    </div>
                    <div className="space-y-4 text-storm-100/60 text-sm leading-relaxed">
                      <p><strong className="text-white">Acceptance:</strong> By accessing or using DocuMind, you agree to be bound by these Terms of Service. If you do not agree, do not use the platform.</p>
                      <p><strong className="text-white">Use License:</strong> DocuMind grants you a limited, non-exclusive, non-transferable license to use the platform for lawful document analysis purposes.</p>
                      <p><strong className="text-white">Prohibited Uses:</strong> You may not use DocuMind to process classified government documents, illegal content, or content that violates intellectual property rights of others without authorization.</p>
                      <p><strong className="text-white">Intellectual Property:</strong> You retain full ownership of all documents you upload. DocuMind does not claim any rights over your content. The platform, its UI, AI models, and branding are owned by DocuMind.</p>
                      <p><strong className="text-white">Limitation of Liability:</strong> DocuMind is provided "as-is" for hackathon demonstration purposes. AI-generated outputs should be verified for accuracy before making critical decisions.</p>
                    </div>
                  </div>
                )}

                {/* Documentation */}
                {footerModal === 'docs' && (
                  <div>
                    <div className="flex items-center gap-3 mb-6">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: '#34d39918' }}>
                        <BookOpen size={20} style={{ color: '#34d399' }} />
                      </div>
                      <h2 className="text-2xl font-bold text-white">Documentation</h2>
                    </div>
                    <div className="space-y-4 text-storm-100/60 text-sm leading-relaxed">
                      <p><strong className="text-white">Getting Started:</strong> Sign in, then drag and drop your documents into the Workspace panel on the left side of the dashboard. Supported formats include PDF, DOCX, PNG, JPG, XLSX, CSV, TXT, and PPTX.</p>
                      <p><strong className="text-white">Document Parsing:</strong> Once uploaded, DocuMind's multi-modal AI engine automatically detects document type, extracts text, tables, charts, and images, and builds a semantic knowledge graph.</p>
                      <p><strong className="text-white">Asking Questions:</strong> Use the central chat panel to ask questions in natural language. You can ask multi-hop questions that span multiple documents. Answers include citation chips linking to source data.</p>
                      <p><strong className="text-white">Source Viewer:</strong> Click any citation chip in the chat to view the extracted source — tables, paragraphs, or images — in the right-side Source Viewer panel.</p>
                      <p><strong className="text-white">Knowledge Graph:</strong> Switch to the "Knowledge Graph" tab in the right panel to visualize semantic relationships between concepts extracted from your documents.</p>
                      <p><strong className="text-white">Tech Stack:</strong> React + Vite frontend, Tailwind CSS for styling, Framer Motion for animations, and Lucide React for icons. Backend uses multi-modal LLM pipelines.</p>
                    </div>
                  </div>
                )}

                {/* Contact */}
                {footerModal === 'contact' && (
                  <div>
                    <div className="flex items-center gap-3 mb-6">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: '#fbbf2418' }}>
                        <Mail size={20} style={{ color: '#fbbf24' }} />
                      </div>
                      <h2 className="text-2xl font-bold text-white">Contact Us</h2>
                    </div>
                    <div className="space-y-5 text-storm-100/60 text-sm leading-relaxed">
                      <p>Have questions, feedback, or want to collaborate? Reach out to the team behind DocuMind.</p>
                      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5">
                        <h4 className="text-white font-semibold mb-3 flex items-center gap-2"><Mail size={16} style={{ color: '#fbbf24' }} /> Email</h4>
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                          <a href="mailto:ayush.kumarverma2024@vitstudent.ac.in" className="text-storm-300 hover:text-storm-100 transition-colors font-mono text-sm sm:text-base underline underline-offset-4 break-all">
                            ayush.kumarverma2024@vitstudent.ac.in
                          </a>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText("ayush.kumarverma2024@vitstudent.ac.in");
                              setCopied(true);
                              setTimeout(() => setCopied(false), 2000);
                            }}
                            className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-white transition-colors text-xs font-medium whitespace-nowrap"
                          >
                            {copied ? <CheckCircle2 size={14} className="text-emerald-400" /> : <Copy size={14} />}
                            {copied ? "Copied" : "Copy to Clipboard"}
                          </button>
                        </div>
                      </div>
                      <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-5">
                        <h4 className="text-white font-semibold mb-3 flex items-center gap-2"><BrainCircuit size={16} style={{ color: '#88BDF2' }} /> Project</h4>
                        <p className="text-storm-100/70">Intelligent Unstructured Document Understanding</p>
                        <p className="text-storm-100/40 text-xs mt-1">Built for Dell Hackathon 2026</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ====== VIDEO DEMO MODAL ====== */}
      <AnimatePresence>
        {showDemo && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4"
            onClick={() => setShowDemo(false)}
          >
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" />

            {/* Modal */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="relative z-10 w-full max-w-4xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Close button */}
              <button
                onClick={() => setShowDemo(false)}
                className="absolute -top-12 right-0 text-white/60 hover:text-white text-sm font-medium flex items-center gap-2 transition-colors"
              >
                Press ESC or click to close ✕
              </button>

              {/* Video container */}
              <div className="relative rounded-2xl overflow-hidden border border-white/10 shadow-2xl shadow-storm-300/10 bg-[#0f1923]">
                {/* Aspect ratio container */}
                <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>

                  {/* ====== INSTRUCTIONS FOR USER: HOW TO ADD VIDEO ====== */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-[#0f1923] to-[#1a2a3a] border-2 border-dashed border-storm-300/30 rounded-2xl m-2">
                    <motion.div
                      animate={{ scale: [1, 1.1, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="w-16 h-16 rounded-full bg-storm-300/20 flex items-center justify-center mb-4"
                    >
                      <Play size={24} className="text-storm-300 ml-1" />
                    </motion.div>
                    <h3 className="text-white text-xl font-bold mb-2">How to add your video:</h3>
                    <p className="text-storm-100/70 text-sm max-w-md text-center mb-4">
                      Open <code className="bg-white/10 px-2 py-1 rounded text-storm-300">App.jsx</code>, go to line ~1000, and replace this instruction block with your video embed code:
                    </p>
                    <div className="bg-black/50 p-4 rounded-lg font-mono text-xs text-storm-300 border border-storm-300/20">
                      {'<iframe'} <br />
                      {'  className="absolute top-0 left-0 w-full h-full"'} <br />
                      {'  src="YOUR_YOUTUBE_OR_VIMEO_LINK_HERE"'} <br />
                      {'  allowFullScreen'} <br />
                      {'/>'}
                    </div>
                  </div>
                  {/* ==================================================== */}

                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Global Styles for Landing */}
      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes shimmer {
          100% { transform: translateX(100%); }
        }
        @keyframes gradient-shift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient-shift {
          background-size: 200% 200%;
          animation: gradient-shift 4s ease infinite;
        }
      `}} />
    </div>
  );
};
// 2. LOGIN PAGE
const LoginPage = ({ setView }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSignIn = (e) => {
    e.preventDefault();
    setError('');

    if (!email) {
      setError('Please enter your email address.');
      return;
    }
    if (!password) {
      setError('Please enter your password.');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    if (!hasUpperCase || !hasLowerCase || !hasNumbers || !hasSpecialChar) {
      setError('Password must contain at least 1 uppercase, 1 lowercase, 1 digit, and 1 special character.');
      return;
    }

    // Validation passed, proceed to dashboard
    setView('dashboard');
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative bg-[#0f1923]">
      {/* Background ambient glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-storm-300/8 blur-[120px] rounded-full pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-md z-10 relative"
      >
        {/* Close / Back button */}
        <button
          onClick={() => setView('landing')}
          className="absolute -top-14 left-0 flex items-center gap-2 text-storm-100/50 hover:text-white text-sm font-medium transition-colors"
        >
          <ArrowLeft size={18} />
          Back to Home
        </button>
        <button
          onClick={() => setView('landing')}
          className="absolute -top-14 right-0 w-10 h-10 rounded-xl bg-white/5 hover:bg-white/10 flex items-center justify-center text-storm-100/50 hover:text-white transition-all"
        >
          <X size={20} />
        </button>

        <GlassCard className="p-8">
          <div className="text-center mb-8">
            <BrainCircuit size={48} className="mx-auto text-storm-300 mb-4" />
            <h2 className="text-3xl font-bold text-white">Welcome Back</h2>
            <p className="text-storm-100/70 mt-2">Sign in to access your multi-modal workspace.</p>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-500/10 border border-red-500/50 text-red-400 px-4 py-3 rounded-lg mb-6 text-sm flex items-start gap-2"
            >
              <Shield size={16} className="mt-0.5 shrink-0" />
              <p>{error}</p>
            </motion.div>
          )}

          <form onSubmit={handleSignIn} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-storm-100 mb-1">Email</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 text-storm-100/50" size={18} />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-storm-900/50 border border-storm-500/30 rounded-lg py-2.5 pl-10 pr-4 text-white focus:outline-none focus:border-storm-300 transition-colors"
                  placeholder="analyst@enterprise.com"
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-sm font-medium text-storm-100">Password</label>
                <button
                  type="button"
                  onClick={() => alert("Forgot password functionality coming soon!")}
                  className="text-xs font-medium text-storm-300 hover:text-white transition-colors"
                >
                  Forgot password?
                </button>
              </div>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-storm-100/50" size={18} />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-storm-900/50 border border-storm-500/30 rounded-lg py-2.5 pl-10 pr-4 text-white focus:outline-none focus:border-storm-300 transition-colors"
                  placeholder="••••••••"
                />
              </div>
            </div>
            <button
              type="submit"
              className="w-full text-[#0f1923] font-bold py-3 rounded-lg mt-6 transition-all shadow-lg hover:shadow-storm-300/25 hover:opacity-90 flex justify-center items-center gap-2"
              style={{ background: 'linear-gradient(90deg, #6A89A7, #88BDF2)' }}
            >
              Sign In <ChevronRight size={18} />
            </button>
          </form>
        </GlassCard>
      </motion.div>
    </div>
  );
};

// 3. MAIN DASHBOARD
const Dashboard = ({ setView }) => {
  const [activeTab, setActiveTab] = useState('source');
  const [activeCitation, setActiveCitation] = useState(null);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]); // Removed mock data for backend integration
  const [docs, setDocs] = useState([]); // Removed mock data for backend integration
  const [showProfile, setShowProfile] = useState(false);
  const fileInputRef = useRef(null);

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages([...messages, { sender: 'user', text: input }]);
    setInput('');
    // Simulate AI typing delay
    setTimeout(() => {
      setMessages(prev => [...prev, {
        sender: 'ai',
        text: 'I am a mock response for the hackathon UI! Processing unstructured graph data...',
        citations: [{ id: 'c3', label: '[Graph Node 42]' }]
      }]);
    }, 1000);
  };

  return (
    <div className="h-screen w-full flex flex-col bg-gradient-to-br from-[#0B1121] via-[#111A2E] to-[#0A192F] text-storm-100 p-2 gap-2 font-sans overflow-hidden relative">

      {/* TOP NAVBAR */}
      <div className="w-full h-14 bg-black/20 rounded-xl border border-white/10 flex items-center justify-between px-4 shrink-0">
        <button
          onClick={() => setView('landing')}
          className="flex items-center gap-2 text-white font-bold hover:text-storm-300 transition-colors"
        >
          <BrainCircuit size={20} className="text-storm-300" />
          DocuMind Home
        </button>

        <div className="relative">
          <button
            onClick={() => setShowProfile(!showProfile)}
            className="w-10 h-10 rounded-full bg-gradient-to-tr from-storm-300/20 to-emerald-400/20 border border-storm-300/30 flex items-center justify-center text-storm-300 font-bold hover:shadow-[0_0_15px_rgba(136,189,242,0.4)] transition-all"
          >
            A
          </button>

          {/* PROFILE DROPDOWN */}
          <AnimatePresence>
            {showProfile && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                className="absolute right-0 top-14 w-64 bg-[#0B1121]/95 backdrop-blur-xl border border-storm-300/20 rounded-xl shadow-2xl overflow-hidden z-50"
              >
                <div className="p-4 border-b border-white/10 bg-white/5">
                  <p className="text-white font-bold text-lg">Analyst User</p>
                  <p className="text-storm-100/60 text-xs font-mono mt-1">analyst@enterprise.com</p>
                </div>
                <div className="p-2">
                  <button
                    onClick={() => setView('landing')}
                    className="w-full text-left px-4 py-2 text-sm font-medium text-red-400 hover:bg-red-400/10 rounded-lg transition-colors flex items-center gap-2"
                  >
                    <X size={16} /> Log Out
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* MAIN 3-PANEL WORKSPACE */}
      <div className="flex-1 w-full flex gap-2 min-h-0">
        {/* LEFT PANEL: Document Management */}
        <GlassCard className="w-1/4 h-full flex flex-col rounded-xl overflow-hidden">
          <div className="p-4 border-b border-white/10 bg-black/10">
            <h2 className="text-lg font-bold flex items-center gap-2 text-white">
              <LayoutDashboard className="text-storm-300" /> Workspace
            </h2>
          </div>
          <div className="p-4 flex-1 overflow-y-auto">
            {/* Drag & Drop Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-storm-500/50 rounded-xl p-6 text-center mb-6 hover:bg-white/5 transition-colors cursor-pointer"
            >
              <UploadCloud className="mx-auto mb-2 text-storm-300" size={32} />
              <p className="text-sm font-medium">Click or Drag & drop documents</p>
              <p className="text-xs text-storm-100/60 mt-1">PDF, DOCX, PNG supported</p>
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept=".pdf,.docx,.png,.jpg,.xlsx,.csv,.txt,.pptx"
                onChange={(e) => {
                  const file = e.target.files[0];
                  if (file) {
                    const ext = file.name.split('.').pop().toLowerCase();
                    const icon = ['png', 'jpg', 'jpeg'].includes(ext)
                      ? <FileImage size={18} />
                      : ['xlsx', 'csv'].includes(ext)
                        ? <FileBarChart size={18} />
                        : <FileText size={18} />;
                    setDocs(prev => [...prev, { id: Date.now(), name: file.name, status: 'Parsed', icon }]);
                  }
                  e.target.value = '';
                }}
              />
            </div>

            <h3 className="text-xs uppercase font-bold text-storm-100/50 tracking-wider mb-3">Ingested Assets</h3>
            <div className="space-y-2">
              {docs.length === 0 ? (
                <div className="text-center p-4 border border-white/5 rounded-lg bg-white/[0.02]">
                  <p className="text-sm text-storm-100/50">No documents uploaded yet.</p>
                </div>
              ) : (
                docs.map(doc => (
                  <div key={doc.id} className="bg-white/5 p-3 rounded-lg flex items-center justify-between group hover:bg-white/10 transition-colors cursor-pointer border border-white/5">
                    <div className="flex items-center gap-3">
                      <div className="text-storm-300">{doc.icon}</div>
                      <div className="truncate w-32 text-sm">{doc.name}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      {doc.status === 'Parsed' ? (
                        <CheckCircle2 size={14} className="text-emerald-400" />
                      ) : (
                        <Loader2 size={14} className="animate-spin text-amber-400" />
                      )}
                      <button
                        onClick={() => setDocs(prev => prev.filter(d => d.id !== doc.id))}
                        className="opacity-0 group-hover:opacity-100 transition-opacity text-storm-100/40 hover:text-red-400"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </GlassCard>

        {/* CENTER PANEL: Chat & QA */}
        <GlassCard className="w-2/4 h-full flex flex-col rounded-xl overflow-hidden relative">
          <div className="p-4 border-b border-white/10 bg-black/10 flex justify-between items-center">
            <h2 className="text-lg font-bold text-white">Multi-Modal Analyst Assistant</h2>
            <span className="text-xs bg-storm-500/20 text-storm-300 px-2 py-1 rounded-full border border-storm-500/30">Llama-3 Reasoning</span>
          </div>

          <div className="flex-1 p-6 overflow-y-auto space-y-6 flex flex-col justify-end">
            {messages.length === 0 ? (
              <div className="text-center text-storm-100/40 my-auto">
                <MessageSquare size={48} className="mx-auto mb-4 opacity-50" />
                <p>No messages yet. Start a conversation to analyze your documents.</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] rounded-2xl p-4 ${msg.sender === 'user' ? 'bg-storm-500 text-white rounded-br-none shadow-lg' : 'bg-white/5 border border-white/10 rounded-bl-none'}`}>
                    <p className="text-sm leading-relaxed">{msg.text}</p>
                    {msg.citations && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {msg.citations.map(cite => (
                          <button
                            key={cite.id}
                            onClick={() => setActiveCitation(cite.id)}
                            className="text-xs bg-storm-900/50 hover:bg-storm-300 hover:text-storm-900 border border-storm-500/40 px-2 py-1 rounded transition-colors flex items-center gap-1"
                          >
                            <Network size={12} /> {cite.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="p-4 bg-black/20 backdrop-blur-md border-t border-white/10">
            <div className="relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask a question across all documents..."
                className="w-full bg-white/5 border border-white/20 rounded-xl py-3 pl-4 pr-12 text-white placeholder-storm-100/40 focus:outline-none focus:border-storm-300"
              />
              <button
                onClick={handleSend}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-storm-500 hover:bg-storm-300 hover:text-storm-900 text-white rounded-lg transition-colors"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </GlassCard>

        {/* RIGHT PANEL: Context Viewer & Graph */}
        <GlassCard className="w-1/4 h-full flex flex-col rounded-xl overflow-hidden">
          <div className="flex border-b border-white/10 bg-black/10">
            <button
              onClick={() => setActiveTab('source')}
              className={`flex-1 p-4 text-sm font-semibold transition-colors ${activeTab === 'source' ? 'text-storm-100 border-b-2 border-storm-300 bg-white/5' : 'text-storm-100/50 hover:text-white'}`}
            >
              Source Viewer
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className={`flex-1 p-4 text-sm font-semibold transition-colors ${activeTab === 'graph' ? 'text-storm-100 border-b-2 border-storm-300 bg-white/5' : 'text-storm-100/50 hover:text-white'}`}
            >
              Knowledge Graph
            </button>
          </div>

          <div className="flex-1 p-4 overflow-y-auto">
            <AnimatePresence mode="wait">
              {activeTab === 'source' ? (
                <motion.div
                  key="source"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  {activeCitation ? (
                    <>
                      <div className="bg-white/5 border border-white/10 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2 text-xs text-storm-300 font-bold uppercase tracking-wider">
                          <FileBarChart size={14} /> Source Data Table
                        </div>
                        <table className="w-full text-xs text-left">
                          <thead>
                            <tr className="border-b border-white/10 text-storm-100/60">
                              <th className="py-2">Division</th>
                              <th className="py-2">Q3 Revenue</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-white/5 bg-storm-500/20">
                              <td className="py-2">Enterprise Cloud</td>
                              <td className="py-2 font-mono text-emerald-400">$4.2M</td>
                            </tr>
                            <tr>
                              <td className="py-2">Hardware</td>
                              <td className="py-2 font-mono">$1.8M</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>

                      <div className="bg-white/5 border border-white/10 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2 text-xs text-storm-300 font-bold uppercase tracking-wider">
                          <FileImage size={14} /> Referenced Image
                        </div>
                        <div className="w-full h-32 bg-storm-900 rounded border border-storm-500/30 flex items-center justify-center relative overflow-hidden">
                          {/* Mock architecture diagram layout */}
                          <div className="w-8 h-8 bg-storm-300 rounded-sm absolute top-4 left-4" />
                          <div className="w-8 h-8 bg-storm-300 rounded-sm absolute top-4 right-4" />
                          <div className="w-12 h-6 bg-emerald-400/80 rounded absolute bottom-4" />
                          <svg className="absolute w-full h-full opacity-30" viewBox="0 0 100 100">
                            <line x1="25" y1="25" x2="50" y2="75" stroke="#fff" strokeWidth="2" />
                            <line x1="75" y1="25" x2="50" y2="75" stroke="#fff" strokeWidth="2" />
                          </svg>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center text-storm-100/40 pt-20">
                      <FileText size={48} className="mb-4 opacity-50" />
                      <p>Click a citation chip in the chat<br />to view extracted multi-modal sources.</p>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="graph"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="h-full flex items-center justify-center relative"
                >
                  {docs.length > 0 ? (
                    <>
                      {/* Mock Knowledge Graph using SVG (Only shows if docs exist) */}
                      <svg className="w-full h-full" viewBox="0 0 200 200">
                        {/* Edges */}
                        <line x1="100" y1="100" x2="50" y2="50" stroke="#6A89A7" strokeWidth="2" strokeDasharray="4" />
                        <line x1="100" y1="100" x2="150" y2="60" stroke="#6A89A7" strokeWidth="2" />
                        <line x1="100" y1="100" x2="120" y2="150" stroke="#6A89A7" strokeWidth="2" />
                        <line x1="150" y1="60" x2="180" y2="90" stroke="#6A89A7" strokeWidth="1" />

                        {/* Nodes */}
                        <circle cx="100" cy="100" r="12" fill="#88BDF2" />
                        <text x="100" y="100" fontSize="6" fill="#000" textAnchor="middle" dy=".3em" fontWeight="bold">Concept</text>

                        <circle cx="50" cy="50" r="10" fill="#6A89A7" />
                        <text x="50" y="35" fontSize="5" fill="#fff" textAnchor="middle">Paragraph 4</text>

                        <circle cx="150" cy="60" r="14" fill="#384959" stroke="#88BDF2" strokeWidth="2" />
                        <text x="150" y="60" fontSize="5" fill="#fff" textAnchor="middle" dy=".3em">Table 3</text>

                        <circle cx="120" cy="150" r="10" fill="#6A89A7" />
                        <text x="120" y="165" fontSize="5" fill="#fff" textAnchor="middle">Image Cap.</text>

                        <circle cx="180" cy="90" r="6" fill="#BDDDFC" />
                      </svg>
                      <div className="absolute bottom-0 text-xs text-storm-100/50 text-center w-full pb-4">
                        Semantic relational mapping visualized.
                      </div>
                    </>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center text-storm-100/40">
                      <Network size={48} className="mb-4 opacity-50" />
                      <p>No knowledge graph generated yet.<br />Upload a document to begin analysis.</p>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </GlassCard>
      </div>
    </div>
  );
};

// --- APP WRAPPER ---
export default function App() {
  const [view, setView] = useState('landing'); // 'landing', 'login', 'dashboard'

  return (
    <>
      {view === 'landing' && <LandingPage setView={setView} />}
      {view === 'login' && <LoginPage setView={setView} />}
      {view === 'dashboard' && <Dashboard setView={setView} />}
    </>
  );
}