"use client";

import React, { useState, useEffect, useRef } from "react";
import { useQuery, useMutation } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { type Id } from "../../../convex/_generated/dataModel";
import {
  Loader2,
  Send,
  User,
  Bot,
  LogIn,
  Sparkles,
  Plus,
  MessageSquare,
  Menu,
  X,
  Trash2,
  Edit2,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

// Type definitions
type Message = {
  role: "user" | "assistant";
  content: string;
};

type Conversation = {
  _id: Id<"conversations">;
  userId: string;
  title: string;
  history: Message[];
  createdAt: number;
  updatedAt: number;
};

export default function ChatbotPage(): React.ReactElement {
  const [userId, setUserId] = useState<string | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isBotReplying, setIsBotReplying] = useState(false);
  const [currentConversationId, setCurrentConversationId] =
    useState<Id<"conversations"> | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [editingTitle, setEditingTitle] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");

  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // --- Authentication ---
  useEffect(() => {
    setTimeout(() => {
      try {
        const storedUserId = localStorage.getItem("userId");
        setUserId(storedUserId);
      } catch (error) {
        console.error("Could not access localStorage:", error);
        setUserId(null);
      } finally {
        setIsAuthLoading(false);
      }
    }, 100);
  }, []);

  // --- Convex Queries & Mutations ---
  const conversations = useQuery(
    api.chat.getUserConversations,
    userId ? { userId } : "skip"
  );

  const currentConversation = useQuery(
    api.chat.getConversationById,
    currentConversationId ? { conversationId: currentConversationId } : "skip"
  );

  const createConversation = useMutation(api.chat.createConversation);
  const updateConversation = useMutation(api.chat.updateConversation);
  const deleteConversation = useMutation(api.chat.deleteConversation);
  const updateConversationTitle = useMutation(api.chat.updateConversationTitle);

  // --- Effects ---
  useEffect(() => {
    if (currentConversation) {
      setMessages(currentConversation.history || []);
    } else if (currentConversationId === null && messages.length === 0) {
      setMessages([
        {
          role: "assistant",
          content:
            "Hello! I'm your AI assistant for gallstone-related questions. How can I help you today? You can ask about symptoms, causes, or treatments.",
        },
      ]);
    }
  }, [currentConversation, currentConversationId]);

  useEffect(() => {
    // Auto-scroll to the latest message
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isBotReplying]);

  // Auto-select first conversation on load
  useEffect(() => {
    if (conversations && conversations.length > 0 && !currentConversationId) {
      setCurrentConversationId(conversations[0]._id);
    }
  }, [conversations, currentConversationId]);

  // --- Helpers ---
  const generateTitle = (firstMessage: string): string => {
    const words = firstMessage.split(" ").slice(0, 6).join(" ");
    return words.length > 30 ? words.substring(0, 30) + "..." : words;
  };

  // --- Event Handlers ---
  const handleNewChat = async () => {
    if (!userId) return;
    try {
      const conversationId = await createConversation({
        userId,
        title: "New Chat",
        history: [],
      });
      setCurrentConversationId(conversationId);
      setMessages([
        {
          role: "assistant",
          content:
            "Hello! I'm your AI assistant for gallstone-related questions. How can I help you today? You can ask about symptoms, causes, or treatments.",
        },
      ]);
    } catch (error) {
      console.error("Failed to create new conversation:", error);
    }
  };

  const handleSelectConversation = (conversationId: Id<"conversations">) => {
    setCurrentConversationId(conversationId);
  };

  const handleDeleteConversation = async (
    conversationId: Id<"conversations">
  ) => {
    try {
      await deleteConversation({ conversationId });
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  };

  const handleTitleEdit = async (
    conversationId: Id<"conversations">,
    title: string
  ) => {
    try {
      await updateConversationTitle({ conversationId, title });
      setEditingTitle(null);
    } catch (error) {
      console.error("Failed to update conversation title:", error);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || !userId || isBotReplying) return;

    const userMessage: Message = { role: "user", content: inputMessage };
    let updatedMessages = [...messages, userMessage];

    // Remove welcome message if it's the only one present before the user's first message
    if (
      updatedMessages.length === 2 &&
      updatedMessages[0].role === "assistant" &&
      updatedMessages[0].content.includes("Hello! I'm your AI assistant")
    ) {
      updatedMessages = [userMessage];
    }

    setMessages(updatedMessages);
    setInputMessage("");
    setIsBotReplying(true);

    try {
      let conversationId = currentConversationId;

      if (!conversationId) {
        const title = generateTitle(inputMessage);
        conversationId = await createConversation({
          userId,
          title,
          history: updatedMessages,
        });
        setCurrentConversationId(conversationId);
      }

      const response = await fetch(
        "https://codexbhawesh-gallstone.hf.space/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: inputMessage }),
        }
      );

      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);

      const data = await response.json();
      const botMessage: Message = { role: "assistant", content: data.response };
      const finalMessages = [...updatedMessages, botMessage];

      setMessages(finalMessages);

      if (conversationId) {
        await updateConversation({
          conversationId,
          history: finalMessages,
        });
      }
    } catch (error) {
      console.error("Failed to get response:", error);
      const errorMessage: Message = {
        role: "assistant",
        content:
          "Sorry, I'm having trouble connecting. Please try again later.",
      };
      const finalMessages = [...updatedMessages, errorMessage];
      setMessages(finalMessages);

      if (currentConversationId) {
        await updateConversation({
          conversationId: currentConversationId,
          history: finalMessages,
        });
      }
    } finally {
      setIsBotReplying(false);
    }
  };

  const handleSignIn = () => {
    window.location.href = "/signin";
  };

  // --- UI Render ---
  if (isAuthLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 text-gray-500">
        <Loader2 className="h-12 w-12 animate-spin text-blue-600 mb-4" />
        <p className="text-xl font-semibold">Authenticating...</p>
      </div>
    );
  }

  if (!userId) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 text-center p-4">
        <div className="bg-white p-8 rounded-2xl shadow-lg max-w-md w-full border border-gray-200">
          <LogIn className="h-16 w-16 text-blue-500 mb-6 mx-auto" />
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            Sign In to Chat
          </h2>
          <p className="text-gray-600 mb-6">
            Please sign in to start a conversation with the AI assistant and
            save your chat history.
          </p>
          <button
            onClick={handleSignIn}
            className="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            <LogIn className="h-5 w-5" />
            Go to Sign In
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-80" : "w-0"
        } bg-gray-900 text-white transition-all duration-300 overflow-hidden flex flex-col`}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Sparkles className="text-blue-400" size={20} />
              Gallstone AI
            </h2>
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-1 hover:bg-gray-700 rounded lg:hidden"
            >
              <X size={20} />
            </button>
          </div>
          <button
            onClick={handleNewChat}
            className="w-full bg-gray-800 hover:bg-gray-700 text-white p-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <Plus size={18} />
            New Chat
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-2">
          {conversations && conversations.length > 0 ? (
            conversations.map((conv) => (
              <div
                key={conv._id}
                className={`group relative mb-2 p-2.5 rounded-lg cursor-pointer transition-colors ${ // FIX: Adjusted padding
                  currentConversationId === conv._id
                    ? "bg-gray-700"
                    : "hover:bg-gray-800"
                }`}
                onClick={() => handleSelectConversation(conv._id)}
              >
                <div className="flex items-center gap-2">
                  <MessageSquare
                    size={16}
                    className="text-gray-400 flex-shrink-0"
                  />
                  {editingTitle === conv._id ? (
                    <input
                      type="text"
                      value={newTitle}
                      onChange={(e) => setNewTitle(e.target.value)}
                      onBlur={() => {
                        handleTitleEdit(conv._id, newTitle);
                        setEditingTitle(null);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          handleTitleEdit(conv._id, newTitle);
                          setEditingTitle(null);
                        } else if (e.key === "Escape") {
                          setEditingTitle(null);
                        }
                      }}
                      className="flex-1 bg-gray-600 text-white text-sm rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      autoFocus
                    />
                  ) : (
                    <span className="text-[13px] truncate flex-1"> {/* FIX: Slightly smaller font */}
                      {conv.title}
                    </span>
                  )}
                </div>
                <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 flex gap-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditingTitle(conv._id);
                      setNewTitle(conv.title);
                    }}
                    className="p-1 hover:bg-gray-600 rounded"
                  >
                    <Edit2 size={14} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteConversation(conv._id);
                    }}
                    className="p-1 hover:bg-gray-600 rounded text-red-400"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="text-gray-400 text-sm p-4 text-center">
              No conversations yet. Start a new chat!
            </div>
          )}
        </div>
      </div>

      {/* --- CHAT AREA (LAYOUT FIX) --- */}
      {/* This container now uses flex-col and overflow-hidden to manage its children's layout and scrolling properly. */}
      <div className="flex-1 flex flex-col relative overflow-hidden">
        {!sidebarOpen && (
          <button
            onClick={() => setSidebarOpen(true)}
            className="absolute top-4 left-4 z-10 p-2 hover:bg-gray-100 rounded-lg"
            aria-label="Open sidebar"
          >
            <Menu size={20} />
          </button>
        )}

        {/* The <main> element is now the single, scrollable container for all messages. */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex items-start gap-3 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                  <Bot size={18} />
                </div>
              )}
              <div
                className={`max-w-3xl p-4 rounded-2xl shadow-sm text-gray-800 ${
                  msg.role === "user"
                    ? "bg-blue-500 text-white rounded-br-md"
                    : "bg-white rounded-bl-md border border-gray-200"
                }`}
              >
                <div className="prose prose-sm max-w-none prose-p:my-2">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
              {msg.role === "user" && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                  <User size={18} />
                </div>
              )}
            </div>
          ))}

          {isBotReplying && (
            <div className="flex items-start gap-3 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                <Bot size={18} />
              </div>
              <div className="max-w-3xl p-4 rounded-2xl bg-white rounded-bl-md border border-gray-200">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse [animation-delay:0.2s]"></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse [animation-delay:0.4s]"></div>
                </div>
              </div>
            </div>
          )}

          {/* This empty div is the target for the auto-scrolling effect. */}
          <div ref={chatEndRef} />
        </main>

        {/* --- INPUT FORM (LAYOUT FIX) --- */}
        {/* This single input form is now placed at the bottom, outside the scrollable <main> area. */}
        {/* It will always remain visible and fixed at the bottom of the chat window. */}
        <div className="bg-gray-50/80 backdrop-blur-sm border-t border-gray-200 p-4">
          <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto">
            <div className="flex items-center gap-3 bg-white rounded-2xl border border-gray-200 p-2 shadow-sm">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask about gallstone symptoms, causes, or treatments..."
                className="flex-1 p-3 bg-transparent focus:outline-none text-gray-700 placeholder-gray-500"
                disabled={isBotReplying}
              />
              <button
                type="submit"
                className="bg-blue-600 text-white w-10 h-10 rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all flex items-center justify-center"
                disabled={!inputMessage.trim() || isBotReplying}
              >
                <Send size={18} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}