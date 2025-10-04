import { useState, useEffect, useRef, Suspense } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Canvas } from "@react-three/fiber";
import { EffectComposer, Bloom } from "@react-three/postprocessing";

import HeadlineTicker from "./components/HeadlineTicker";
import NewsFeed from "./components/NewsFeed";
import MediaPanel from "./components/MediaPanel";
import ScanPanel from "./components/ScanPanel";
import ChatBox from "./components/ChatBox";
import PhishingChallenge from "./components/PhishingChallenge";
import PhishRadar from "./components/PhishRadar";

gsap.registerPlugin(ScrollTrigger);

function Navbar() {
  return (
    <nav className="fixed top-0 left-0 w-full z-50 bg-black/60 backdrop-blur-md shadow-md">
      <div className="max-w-6xl mx-auto flex items-center justify-between p-4">
        <div className="flex items-center gap-3">
          <img src="root/public/logo.png" alt="Crypt Chat Logo" className="w-26.5 h-10 rounded" />
       
        </div>
        <ul className="hidden sm:flex items-center gap-6 text-gray-300 font-medium">
          <li><a href="#news" className="hover:text-green-400">News</a></li>
          <li><a href="#scan" className="hover:text-green-400">Scan</a></li>
          <li><a href="#challenge" className="hover:text-green-400">Challenge</a></li>
          <li><a href="#radar" className="hover:text-green-400">Radar</a></li>
        </ul>
      </div>
    </nav>
  );
}

function BackgroundMesh() {
  return (
    <mesh rotation={[0.2, 0.5, 0]} position={[0, 0, -2]}>
      <icosahedronGeometry args={[3.8, 3]} />
      <meshStandardMaterial
        color="#001f2a"
        emissive="#063447"
        emissiveIntensity={0.8}
        roughness={0.45}
        metalness={0.2}
      />
    </mesh>
  );
}

// Simple phishing interceptor
async function checkPhish({ url, file, image }) {
  try {
    const formData = new FormData();
    if (url) formData.append("url", url);
    if (file) formData.append("file", file);
    if (image) formData.append("image", image);

    const res = await fetch("/api/check-phish", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    return data;
  } catch (err) {
    console.error("Phishing check failed:", err);
    return { phishing: false };
  }
}

function HeroCanvas() {
  const canvasRef = useRef();
  const sectionRef = useRef();
  const frameCount = 425;
  const images = useRef([]);

  useEffect(() => {
    for (let i = 1; i <= frameCount; i++) {
      const img = new Image();
      img.src = `root/public/frame_${String(i).padStart(4, "0")}.jpeg`;
      images.current.push(img);
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const render = (frame) => {
      const img = images.current[frame];
      if (!img) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    render(0);

    let frame = 0;
    const intervalId = setInterval(() => {
      frame = (frame + 1) % frameCount;
      render(frame);
    }, 40);

    ScrollTrigger.create({
      trigger: sectionRef.current,
      start: "top top",
      end: `+=${frameCount * 10}`,
      pin: true,
      pinSpacing: true
    });

    return () => {
      clearInterval(intervalId);
      window.removeEventListener("resize", resizeCanvas);
      ScrollTrigger.kill();
    };
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-[100vh] overflow-hidden pt-16"
    >
      <canvas ref={canvasRef} className="absolute inset-0" />
      <div className="absolute inset-0 pointer-events-none z-10">
        <h2 style={{ top: "15vh" }} className="absolute text-3xl font-bold opacity-80 text-center w-full">
          🔐 End-to-End Encrypted Chats
        </h2>
        <h2 style={{ top: "120vh" }} className="absolute text-3xl font-bold opacity-80 text-center w-full">
          🤖 AI PhishGuard Assistant
        </h2>
        <h2 style={{ top: "240vh" }} className="absolute text-3xl font-bold opacity-80 text-center w-full">
          🌍 Global PhishRadar Map
        </h2>
        <h2 style={{ top: "360vh" }} className="absolute text-3xl font-bold opacity-80 text-center w-full">
          🎮 Gamified Security Challenges
        </h2>
      </div>
    </section>
  );
}

export default function App() {
  const chatRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [phishAlert, setPhishAlert] = useState(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!isLoading && chatRef.current) {
      gsap.to(chatRef.current, {
        boxShadow: "0 0 30px 6px rgba(20,241,149,0.12)",
        repeat: -1,
        yoyo: true,
        duration: 2,
        ease: "sine.inOut"
      });
    }
  }, [isLoading]);

  if (isLoading)
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-black text-green-400 space-y-6">
        <div className="w-20 h-20 border-4 border-green-400 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-xl font-bold animate-pulse">🔐 Encrypting Messages...</p>
      </div>
    );

  return (
    <div className="min-h-screen relative overflow-x-hidden font-sans bg-bgDark text-gray-200">
      <Navbar />

      {phishAlert && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 bg-red-600 text-white px-6 py-3 rounded-lg z-50 shadow-lg">
          {phishAlert}
        </div>
      )}

      <HeroCanvas />

      <div className="relative z-20 -mt-24">
        <HeadlineTicker />
      </div>

      <main className="max-w-6xl mx-auto p-6 space-y-16 relative z-10">
        <section id="news">
          <NewsFeed />
        </section>
        <MediaPanel />
        <section id="scan" className="space-y-6">
          <h3 className="text-2xl font-semibold">Scan a file or URL</h3>
          <ScanPanel checkPhish={checkPhish} setPhishAlert={setPhishAlert} />
        </section>
        <section id="challenge" className="space-y-6">
          <h3 className="text-2xl font-semibold">Phishing Challenge & Gamification</h3>
          <PhishingChallenge />
        </section>
        <section id="radar" className="space-y-6">
          <h3 className="text-2xl font-semibold">PhishRadar (Demo)</h3>
          <PhishRadar />
        </section>
      </main>

      <ChatBox refProp={chatRef} />

      <footer className="max-w-6xl mx-auto p-6 text-sm text-gray-500 relative z-10">
        <div className="flex items-center justify-between">
          <div>© Crypt Chat — Privacy-first demo</div>
          <div>Inspired by VirusTotal, SonicLamb, Truecaller, ChitChat.</div>
        </div>
      </footer>

      <Canvas className="fixed inset-0 -z-10" camera={{ position: [0, 0, 8], fov: 55 }}>
        <color attach="background" args={["#001217"]} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <Suspense fallback={null}>
          <BackgroundMesh />
          <EffectComposer>
            <Bloom luminanceThreshold={0.25} luminanceSmoothing={0.7} height={250} />
          </EffectComposer>
        </Suspense>
      </Canvas>
    </div>
  );
}
