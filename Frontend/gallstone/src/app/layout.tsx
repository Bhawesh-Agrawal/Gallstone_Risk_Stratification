import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Providers from "./providers";
import Navbar from "./Components/Navbar/page";
import Footer from "./Components/Footer/page";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const baseUrl = "https://gallitify.bhaweshagrawal.com.np";

export const metadata: Metadata = {
  // General App Metadata
  title: {
    template: "%s | Gallitify",
    default: "Gallitify | Wellness for Everyone", 
  },
  description: "Using Modern Technology to provide best healtcare",

  icons: {
    icon: "/logo.png",
  },

  openGraph: {
    title: "Gallitify | Wellness for Everyone",
    description: "Using Modern Technology to provide best healthcare",
    url: baseUrl,
    siteName: "Galllitify",
    images: [
      {
        url: `${baseUrl}/Social Media.png`, // IMPORTANT: Create a dedicated social sharing image (1200x630px) and place it in your /public folder
        width: 1200,
        height: 630,
        alt: "Using Modern Technology to provide best healthcare",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  
  twitter: {
    card: "summary_large_image",
    title: "Gallitify | Wellness for Everyone",
    description: "Using Modern Technology to provide best healthcare",
    site: "@BhaweshAgr87299",
    creator: "@BhaweshAgr87299",
    images: [`${baseUrl}/Social Media.png`], 
  },

  metadataBase: new URL(baseUrl),
  alternates: {
    canonical: '/', 
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Providers>
          {/* Navbar at the top */}
          <Navbar />

          {/* Main content */}
          <main>{children}</main>

          {/* Footer at the bottom */}
          <Footer />
        </Providers>
      </body>
    </html>
  );
}