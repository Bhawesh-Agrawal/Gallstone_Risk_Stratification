"use client";
import React from "react";
import Image from "next/image";
import { motion } from "framer-motion";

export default function LandingPage() {
  return (
    <div className="relative bg-gradient-to-r from-blue-900 to-indigo-900 text-white overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-full">
            <div className="absolute top-8 left-1/4 w-4 h-4 bg-red-500 transform rotate-45 opacity-30"></div>
            <div className="absolute bottom-16 right-1/4 w-6 h-6 bg-blue-500 rounded-full opacity-30"></div>
            <div className="absolute top-1/3 right-10 w-8 h-8 border-2 border-green-400 transform rotate-12 opacity-30"></div>
            <div className="absolute bottom-1/4 left-10 w-10 h-10 border-2 border-purple-400 rounded-full opacity-30"></div>
            <div className="absolute top-1/4 left-[80px] w-2 h-2 bg-red-700 opacity-50 rounded-full"></div>
            <div className="absolute bottom-1/3 right-[120px] w-3 h-3 bg-teal-300 opacity-50 transform rotate-90"></div>
      </div>
      {/* Main Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-12 py-16 lg:py-28 flex flex-col lg:flex-row items-center lg:justify-between">
        {/* Left Section */}
        <motion.div
          className="lg:w-1/2 text-center lg:text-left mb-12 lg:mb-0"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          viewport={{ once: true }}
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight mb-6">
            <span className="text-blue-300">Know Your Health</span> Before It
            Happens.
          </h1>
          <p className="text-base sm:text-lg text-gray-300 mb-8 max-w-lg mx-auto lg:mx-0">
            Our cutting-edge model predicts the risk of gallstones with{" "}
            <span className="text-white font-semibold">up to 95% accuracy</span>.
          </p>
          <motion.button
            className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-8 rounded-full text-lg transition duration-300 ease-in-out shadow-lg"
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.95 }}
          >
            Explore Our Predictive Service
          </motion.button>
        </motion.div>

        {/* Right Section (Droplet Image) */}
        <motion.div
          className="lg:w-1/2 flex justify-center lg:justify-end relative"
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <motion.div
            className="relative w-[260px] sm:w-[300px] lg:w-[360px] h-[420px] sm:h-[480px] lg:h-[520px]"
            animate={{ y: [0, -12, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          >
            {/* SVG droplet with image clipped inside */}
            <svg
              viewBox="0 0 220 300"
              className="w-full h-full drop-shadow-2xl"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Define the clipPath (droplet shape) */}
              <defs>
                <clipPath id="dropletClip" clipPathUnits="objectBoundingBox">
                  <path
                    d="M0.5,0 
                       C0.72,0.3,0.95,0.57,0.86,0.8 
                       C0.79,0.96,0.21,0.96,0.14,0.8 
                       C0.05,0.57,0.28,0.3,0.5,0 Z"
                    transform="scale(1,1)"
                  />
                </clipPath>
                <linearGradient id="dropletGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.85" />
                  <stop offset="100%" stopColor="#1e40af" stopOpacity="0.95" />
                </linearGradient>
              </defs>

              {/* Background droplet */}
              <path
                d="M110 0 
                   C160 90 210 170 190 240 
                   C175 290 45 290 30 240 
                   C10 170 60 90 110 0 Z"
                fill="url(#dropletGradient)"
              />

              {/* Image inside droplet */}
              <foreignObject
                x="0"
                y="0"
                width="220"
                height="300"
                clipPath="url(#dropletClip)"
              >
                <div className="w-full h-full">
                  <div className="absolute inset-0 bg-blue-700 rounded-full transform scale-125 translate-x-10 -translate-y-10 opacity-30 filter blur-3xl"></div>
                  <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center"></div>
                  <Image
                    src="https://plus.unsplash.com/premium_photo-1661580574627-9211124e5c3f?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                    alt="Doctor and Patient"
                    width={220}
                    height={300}
                    className="object-cover w-full h-full"
                  />
                <div className="absolute bottom-10 left-0 lg:-left-10 w-24 h-12 bg-gray-700 rounded-full flex items-center justify-center transform rotate-12 opacity-70 hidden sm:flex">
                    <div className="w-10 h-6 bg-red-500 rounded-l-full"></div>
                    <div className="w-10 h-6 bg-blue-500 rounded-r-full"></div>
                </div>
                </div>
              </foreignObject>
            </svg>
          </motion.div>
        </motion.div>
      </div>
      <div className="absolute bottom-0 left-0 w-full overflow-hidden leading-[0] rotate-180">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 1200 120"
          preserveAspectRatio="none"
          className="relative block w-full h-[100px] sm:h-[120px]"
        >
          <path
            d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,
            82.39-16.72,168.19-17.73,250.45-.39
            C823.78,31,906.67,72,985.66,92.83
            c70.05,18.48,146.53,26.09,214.34,3V0H0
            V27.35A600.21,600.21,0,0,0,321.39,56.44Z"
            fill="#ffffff"
          />
        </svg>
      </div>
    </div>
  );
}
