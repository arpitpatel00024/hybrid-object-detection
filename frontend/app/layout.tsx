import './globals.css';
import { ReactNode } from 'react';

export const metadata = {
  title: 'Hybrid YOLO + Fuzzy Object Detection',
  description: 'Real-time Object Detection with Fuzzy Priority Evaluation',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col items-center justify-start p-6">
        <header className="w-full max-w-6xl mb-8 border-b border-slate-700 pb-4">
          <h1 className="text-3xl font-bold text-sky-400">Hybrid Detection Engine</h1>
          <p className="text-slate-400 text-sm mt-1">
            YOLOv8 Detection + Fuzzy Logic Priority System + Genetic Optimization
          </p>
        </header>
        <main className="w-full max-w-6xl">{children}</main>
      </body>
    </html>
  );
}
