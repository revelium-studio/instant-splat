import type { Metadata } from "next";
import AnalyticsWrapper from "@/components/AnalyticsWrapper";
import "./globals.css";

export const metadata: Metadata = {
  title: "Revelium Studio | InstantSplat++",
  description: "Transform any photograph into an explorable 3D scene with InstantSplat++",
  icons: {
    icon: [
      {
        url: "https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/favicon_black.png",
        media: "(prefers-color-scheme: light)",
      },
      {
        url: "https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/favicon_white.png",
        media: "(prefers-color-scheme: dark)",
      },
    ],
    shortcut: "https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/favicon_black.png",
    apple: "https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/favicon_black.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
        <AnalyticsWrapper />
      </body>
    </html>
  );
}
