"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  User,
  Menu,
  X,
  MessageCircle,
  FileText,
  LogIn,
  UserPlus,
  Home,
  ChevronDown,
  LogOut,
  History,
} from "lucide-react";
import { api } from "../../../../convex/_generated/api";
import { useQuery } from "convex/react";
import { Id } from "../../../../convex/_generated/dataModel";

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [userId, setUserId] = useState<Id<"users"> | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedId = localStorage.getItem("userId");
      setUserId(storedId as Id<"users"> | null);
      setIsInitialized(true);
    }
  }, []);

  const userData = useQuery(
    api.task.getUserById,
    userId ? { userId } : "skip"
  );

  const isLoggedIn = useMemo(() => Boolean(userData), [userData]);
  const isLoading = useMemo(
    () => !isInitialized || (userId && userData === undefined),
    [isInitialized, userId, userData]
  );

  const navLinks = useMemo(
    () => [
      { href: "/", label: "Home", icon: Home },
      { href: "/predict", label: "Predict", icon: FileText },
      { href: "/chatbot", label: "AI Assistant", icon: MessageCircle },
    ],
    []
  );

  const handleLogout = useCallback(() => {
    setUserId(null);
    setIsDropdownOpen(false);
    setIsMenuOpen(false);
    localStorage.removeItem("userId");
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isDropdownOpen || isMenuOpen) {
        const target = event.target as HTMLElement;
        if (
          !target.closest("[data-dropdown]") &&
          !target.closest("[data-mobile-menu]")
        ) {
          setIsDropdownOpen(false);
          setIsMenuOpen(false);
        }
      }
    };

    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, [isDropdownOpen, isMenuOpen]);

  if (isLoading) {
    return (
      <nav className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Updated Logo for Loading State */}
            <a href="/" className="flex items-center space-x-2">
              <img
                src="/logo.png"
                alt="Gallitify Logo"
                className="h-10 w-auto"
              />
              <span className="text-2xl font-bold text-gray-800">
                Galli<span className="text-blue-600">tify</span>
              </span>
            </a>
            <div className="hidden md:flex items-center space-x-4">
              <div className="w-20 h-4 bg-gray-200 rounded animate-pulse" />
              <div className="w-16 h-4 bg-gray-200 rounded animate-pulse" />
            </div>
          </div>
        </div>
      </nav>
    );
  }

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Updated Logo */}
          <a href="/" className="flex items-center space-x-2">
            <img
              src="/logo.png"
              alt="Gallitify Logo"
              className="h-10 w-auto"
            />
            <span className="text-2xl font-bold text-gray-800">
              Galli<span className="text-blue-600">tify</span>
            </span>
          </a>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="flex items-center space-x-1 text-gray-700 hover:text-blue-600 transition-colors duration-200"
              >
                <link.icon className="h-4 w-4" />
                <span>{link.label}</span>
              </a>
            ))}
          </div>

          {/* Desktop Auth Section */}
          <div className="hidden md:flex items-center space-x-4">
            {isLoggedIn && userData ? (
              <div className="relative" data-dropdown>
                <button
                  onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                  className="flex items-center space-x-2 p-1 rounded-full hover:bg-gray-100 transition-colors duration-200"
                  aria-expanded={isDropdownOpen}
                >
                  {/* Profile Photo or Default */}
                  {userData.photo_link && userData.photo_link.trim() !== "" ? (
                    <img
                      src={userData.photo_link}
                      alt={userData.name}
                      className="w-8 h-8 rounded-full object-cover border"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src =
                          "https://via.placeholder.com/150?text=No+Photo";
                      }}
                    />
                  ) : (
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <User className="h-4 w-4 text-blue-600" />
                    </div>
                  )}

                  <span className="text-sm font-medium text-gray-700 capitalize">
                    {userData.name}
                  </span>
                  <ChevronDown
                    className={`h-4 w-4 text-gray-500 transition-transform duration-200 ${
                      isDropdownOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>

                {isDropdownOpen && (
                  <div className="absolute right-0 mt-2 w-56 bg-white rounded-md shadow-lg py-1 border z-50 animate-in fade-in-0 zoom-in-95">
                    <a
                      href="/profile"
                      className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-blue-50 transition-colors duration-200"
                    >
                      <User className="h-4 w-4" />
                      <span>Profile</span>
                    </a>

                    <a
                      href="/patient-history"
                      className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-blue-50 transition-colors duration-200"
                    >
                      <History className="h-4 w-4" />
                      <span>Patient History</span>
                    </a>

                    <button
                      onClick={handleLogout}
                      className="w-full text-left flex items-center space-x-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors duration-200"
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Sign Out</span>
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <>
                <a
                  href="/signin"
                  className="flex items-center space-x-1 text-gray-700 hover:text-blue-600 transition-colors duration-200"
                >
                  <LogIn className="h-4 w-4" />
                  <span>Sign In</span>
                </a>
                <a
                  href="/createaccount"
                  className="flex items-center space-x-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200"
                >
                  <UserPlus className="h-4 w-4" />
                  <span>Sign Up</span>
                </a>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-gray-700 hover:text-blue-600 p-2 transition-colors duration-200"
              aria-expanded={isMenuOpen}
            >
              {isMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMenuOpen && (
          <div
            className="md:hidden border-t animate-in slide-in-from-top-2"
            data-mobile-menu
          >
            <div className="px-2 pt-2 pb-3 space-y-1">
              {navLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="flex items-center space-x-2 px-3 py-2 text-gray-700 hover:bg-blue-50 rounded-md transition-colors duration-200"
                >
                  <link.icon className="h-4 w-4" />
                  <span>{link.label}</span>
                </a>
              ))}
              <div className="border-t pt-4 mt-2">
                {isLoggedIn && userData ? (
                  <div className="space-y-2">
                    <a
                      href="/profile"
                      className="flex items-center space-x-2 px-3 py-2 text-gray-700 hover:bg-blue-50 rounded-md transition-colors duration-200"
                    >
                      <User className="h-4 w-4" />
                      <span>Profile</span>
                    </a>

                    <a
                      href="/patient-history"
                      className="flex items-center space-x-2 px-3 py-2 text-gray-700 hover:bg-blue-50 rounded-md transition-colors duration-200"
                    >
                      <History className="h-4 w-4" />
                      <span>Patient History</span>
                    </a>

                    <button
                      onClick={handleLogout}
                      className="w-full text-left flex items-center space-x-2 px-3 py-2 text-red-600 hover:bg-red-50 rounded-md transition-colors duration-200"
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Sign Out</span>
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <a
                      href="/signin"
                      className="flex items-center justify-center w-full px-3 py-2 text-gray-700 hover:bg-blue-50 rounded-md transition-colors duration-200"
                    >
                      <LogIn className="h-4 w-4 mr-2" />
                      <span>Sign In</span>
                    </a>
                    <a
                      href="/createaccount"
                      className="flex items-center justify-center w-full px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors duration-200"
                    >
                      <UserPlus className="h-4 w-4 mr-2" />
                      <span>Sign Up</span>
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}