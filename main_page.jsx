// App.jsx
import React from "react";
import { motion } from "framer-motion";

function App() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50 text-gray-800 font-sans">
      {/* Header */}
      <header className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-green-600">GSmart</h1>
          <nav className="space-x-6 hidden md:block">
            <a href="#about" className="hover:text-green-600 transition">About</a>
            <a href="#services" className="hover:text-green-600 transition">Services</a>
            <a href="#contact" className="hover:text-green-600 transition">Contact</a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
        className="flex-1 flex items-center justify-center bg-gradient-to-r from-green-600 to-emerald-500 text-white"
      >
        <div className="max-w-3xl text-center px-6">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Welcome to <span className="text-yellow-300">GSmart</span>
          </h2>
          <p className="text-lg md:text-xl mb-8">
            Smart solutions for a connected world.  
            Technology that simplifies life and empowers businesses.
          </p>
          <motion.a
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            href="#about"
            className="bg-yellow-300 text-gray-900 font-semibold px-6 py-3 rounded-lg shadow hover:bg-yellow-400 transition"
          >
            Learn More
          </motion.a>
        </div>
      </motion.section>

      {/* About Section */}
      <motion.section
        id="about"
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="py-16 px-6 bg-white"
      >
        <div className="max-w-5xl mx-auto text-center">
          <h3 className="text-3xl font-bold text-green-600 mb-6">About GSmart</h3>
          <p className="text-lg text-gray-600 leading-relaxed">
            GSmart is a forward-thinking company delivering innovative digital solutions.  
            Our mission is to bridge the gap between technology and people by offering reliable, scalable, 
            and impactful products that truly make a difference.
          </p>
        </div>
      </motion.section>

      {/* Services Section */}
      <motion.section
        id="services"
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="py-16 px-6 bg-gray-100"
      >
        <div className="max-w-6xl mx-auto">
          <h3 className="text-3xl font-bold text-green-600 text-center mb-10">Our Services</h3>
          <div className="grid gap-8 md:grid-cols-3">
            {[
              { title: "Smart Applications", desc: "Modern web & mobile apps designed for performance, scalability, and simplicity." },
              { title: "Data Solutions", desc: "Transforming raw data into actionable insights with AI, analytics, and cloud solutions." },
              { title: "Consulting", desc: "Helping businesses adopt cutting-edge technology with expert guidance and strategy." }
            ].map((service, idx) => (
              <motion.div
                key={idx}
                whileHover={{ scale: 1.05 }}
                className="bg-white p-6 rounded-lg shadow hover:shadow-xl transition cursor-pointer"
              >
                <h4 className="text-xl font-semibold mb-3 text-green-600">{service.title}</h4>
                <p className="text-gray-600">{service.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Contact Section */}
      <motion.section
        id="contact"
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="py-16 px-6 bg-white"
      >
        <div className="max-w-4xl mx-auto text-center">
          <h3 className="text-3xl font-bold text-green-600 mb-6">Get in Touch</h3>
          <p className="text-gray-600 mb-8">
            Have questions or want to work with us? Reach out today and let’s build the future together.
          </p>
          <motion.a
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            href="mailto:contact@gsmart.com"
            className="bg-green-600 text-white font-semibold px-6 py-3 rounded-lg shadow hover:bg-green-700 transition"
          >
            Contact Us
          </motion.a>
        </div>
      </motion.section>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-6 text-center">
        <p>&copy; {new Date().getFullYear()} GSmart. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
