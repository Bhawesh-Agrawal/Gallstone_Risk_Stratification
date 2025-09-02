"use client";

import React from "react";
import Link from "next/link";

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-300 py-6 mt-16">
      <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row justify-between items-center">
        
        {/* Links */}
        <div className="flex space-x-6 mb-4 sm:mb-0">
          <Link href="/predict" className="hover:text-white transition-colors">
            Predict
          </Link>
          <Link href="/chatbot" className="hover:text-white transition-colors">
            Chatbot
          </Link>
        </div>

        {/* Trademark */}
        <p className="text-sm text-gray-400">
          © {new Date().getFullYear()} Gallitify™. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
