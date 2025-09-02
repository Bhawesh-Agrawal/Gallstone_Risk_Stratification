"use client";

import React from "react";
import { Star, Quote } from "lucide-react";

export default function Testimonials() {
  const reviews = [
    {
      name: "Sophia Johnson",
      role: "Patient",
      feedback:
        "Gallitify helped me detect gallstones early before symptoms got worse. The predictive insights gave me peace of mind and a clear action plan.",
      rating: 5,
    },
    {
      name: "Dr. Michael Carter",
      role: "Gastroenterologist",
      feedback:
        "As a doctor, I find Gallitify’s AI predictions reliable and accurate. It helps me support my patients with data-driven decisions much faster.",
      rating: 5,
    },
    {
      name: "Rajesh Patel",
      role: "Patient",
      feedback:
        "I was able to take preventive steps thanks to Gallitify’s early risk alerts. The platform is easy to use and incredibly helpful.",
      rating: 4,
    },
  ];

  return (
    <section className="bg-white py-20 relative">
      <div className="max-w-7xl mx-auto px-6 lg:px-12 text-center">
        {/* Section Header */}
        <h3 className="text-blue-600 font-semibold uppercase tracking-wide mb-2">
          Testimonials
        </h3>
        <h2 className="text-3xl lg:text-4xl font-extrabold text-gray-900 leading-tight mb-10">
          What People Say About <span className="text-blue-700">Gallitify</span>
        </h2>

        {/* Reviews Grid */}
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          {reviews.map((review, index) => (
            <div
              key={index}
              className="bg-gray-50 rounded-2xl shadow-lg p-6 relative hover:scale-105 transition-transform duration-300"
            >
              {/* Quote Icon */}
              <div className="absolute -top-6 left-6 bg-blue-600 p-3 rounded-full shadow-md">
                <Quote className="w-5 h-5 text-white" />
              </div>

              {/* Feedback */}
              <p className="text-gray-700 text-base mb-4 leading-relaxed">
                “{review.feedback}”
              </p>

              {/* Rating */}
              <div className="flex gap-1 justify-center mb-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Star
                    key={i}
                    className={`w-5 h-5 ${
                      i < review.rating
                        ? "text-yellow-400 fill-yellow-400"
                        : "text-gray-300"
                    }`}
                  />
                ))}
              </div>

              {/* Name & Role */}
              <div>
                <h4 className="font-semibold text-gray-900">
                  {review.name}
                </h4>
                <p className="text-sm text-blue-600">{review.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
