// src/app/api/hello/route.js

export async function GET(request) {
    return new Response(JSON.stringify({ message: 'Hello, World!' }), {
      headers: { 'Content-Type': 'application/json' },
    });
}
  