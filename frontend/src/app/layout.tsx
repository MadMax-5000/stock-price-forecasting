import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  variable: "--font-matter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Prédicteur de prix d'actions",
  description: "Prédiction du prix des actions借助机器学习",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr" className={outfit.variable}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
