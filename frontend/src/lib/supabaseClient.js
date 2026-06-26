import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// Initialize the Supabase client safely
let supabaseInstance = null;

try {
  // Only initialize if the URL looks like a valid HTTP URL
  if (supabaseUrl && supabaseUrl.startsWith('http') && supabaseAnonKey) {
    supabaseInstance = createClient(supabaseUrl, supabaseAnonKey);
  }
} catch (error) {
  console.error("Failed to initialize Supabase client:", error);
}

export const supabase = supabaseInstance;

if (!supabase) {
  console.warn("Supabase credentials not found or invalid. Authentication will fail. Please add a valid VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to your .env file.");
}
