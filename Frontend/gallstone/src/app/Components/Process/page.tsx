"use client";

import React from "react";
import Image from "next/image";

export default function Process() {
  const steps = [
    {
      number: "01",
      title: "Fill the Form",
      desc: "Provide your health details in our secure form to start the prediction process.",
      img: "https://plus.unsplash.com/premium_photo-1661430678268-3a855fd087b1?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    },
    {
      number: "02",
      title: "Predict & Get Report",
      desc: "Our AI model analyzes your inputs and instantly generates a detailed health report.",
      img: "https://images.unsplash.com/photo-1618044733300-9472054094ee?q=80&w=1171&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    },
    {
      number: "03",
      title: "Ask Anything to Chatbot",
      desc: "Get personalized answers and guidance from our AI-powered medical chatbot anytime.",
      img: "https://images.unsplash.com/photo-1659018966834-99be9746b49e?q=80&w=1198&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    },
  ];

  return (
    <section className="bg-gray-50 py-20 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-6 lg:px-12 text-center">
        {/* Header */}
        <h3 className="text-blue-600 font-semibold uppercase tracking-wide mb-2">
          Process
        </h3>
        <h2 className="text-3xl lg:text-4xl font-extrabold text-gray-900 mb-14">
          How <span className="text-blue-700">Gallitify</span> Helps You Step by Step
        </h2>

        {/* Steps Grid */}
        <div className="grid md:grid-cols-3 gap-12 relative">
          {steps.map((step, i) => (
            <div key={i} className="relative group">
              {/* Number in Background */}
              <span className="absolute -top-8 left-1/2 -translate-x-1/2 text-6xl font-extrabold text-gray-200 opacity-60">
                {step.number}
              </span>

              {/* Card */}
              <div className="bg-white rounded-2xl shadow-lg p-6 flex flex-col items-center text-center relative z-10 hover:shadow-2xl transition">
                <div className="w-full h-48 rounded-xl overflow-hidden mb-6">
                  <Image
                    src={step.img}
                    alt={step.title}
                    width={400}
                    height={300}
                    className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-500"
                  />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 mb-2">
                  {step.title}
                </h4>
                <p className="text-gray-600 text-sm">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
