"use client";

import React from "react";
import Image from "next/image";
import { CheckCircle } from "lucide-react";

export default function AboutUs() {
  return (
    <section className="bg-gray-50 py-20 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-6 lg:px-12 grid lg:grid-cols-2 gap-12 items-center">
        {/* Left: Image + floating cards */}
        <div className="relative flex justify-center items-center">
          <div className="relative w-full max-w-md h-[450px] rounded-[40px] overflow-hidden shadow-2xl border border-gray-200">
            <Image
              src="https://plus.unsplash.com/premium_photo-1673953509975-576678fa6710?q=80&w=1170&auto=format&fit=crop"
              alt="About Gallitify"
              fill
              className="object-cover hover:scale-105 transition-transform duration-500 ease-in-out"
            />
          </div>

          {/* Floating Card (Name/Role) */}
          <div className="absolute bottom-8 -left-6 bg-white px-6 py-3 rounded-2xl shadow-lg">
            <h4 className="font-semibold text-gray-800">AI Powered</h4>
            <p className="text-sm text-red-500">Medical Assistant</p>
          </div>

          {/* Floating Icon */}
          <div className="absolute top-6 right-6 bg-blue-600 p-4 rounded-2xl shadow-lg">
            <span className="text-white text-2xl">❤</span>
          </div>
        </div>

        {/* Right: Text Content */}
        <div>
          <h3 className="text-blue-600 font-semibold mb-2">About Us</h3>
          <h2 className="text-3xl lg:text-4xl font-extrabold text-gray-900 leading-tight mb-4">
            Empowering Health Through Predictive Insights
          </h2>
          <p className="text-gray-600 text-lg mb-6">
            At <span className="font-semibold text-blue-700">Gallitify</span>,
            we believe in a future where health challenges are anticipated, not
            just reacted to. Our platform harnesses the power of advanced AI to
            provide early, accurate predictions for conditions like gallstones,
            giving individuals and healthcare providers the knowledge to act
            proactively.
          </p>

          {/* Checklist */}
          <ul className="grid sm:grid-cols-2 gap-3 text-gray-700 mb-8">
            {[
              "AI-Powered Predictions",
              "Early Risk Detection",
              "Personalized Insights",
              "Proactive Healthcare",
            ].map((item, i) => (
              <li key={i} className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-blue-600" />
                {item}
              </li>
            ))}
          </ul>

        </div>
      </div>
    </section>
  );
}
