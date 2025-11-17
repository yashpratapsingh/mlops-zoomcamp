import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [customers, setCustomers] = useState([]);
  const [form, setForm] = useState({ name: "", email: "", city: "" });
  const [editing, setEditing] = useState(null);

  const fetchCustomers = async () => {
    const res = await axios.get(`${API_BASE}/customers`);
    setCustomers(res.data);
  };

  useEffect(() => {
    fetchCustomers();
  }, []);

  const createCustomer = async () => {
    await axios.post(`${API_BASE}/customers`, form);
    setForm({ name: "", email: "", city: "" });
    fetchCustomers();
  };

  const updateCustomer = async (id) => {
    await axios.put(`${API_BASE}/customers/${id}`, form);
    setEditing(null);
    setForm({ name: "", email: "", city: "" });
    fetchCustomers();
  };

  const deleteCustomer = async (id) => {
    await axios.delete(`${API_BASE}/customers/${id}`);
    fetchCustomers();
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Lakebase CRUD UI</h1>

      <div className="bg-white shadow p-6 rounded-lg mb-6">
        <h2 className="text-xl font-semibold mb-4">{editing ? "Edit" : "Create"} Customer</h2>
        <div className="flex gap-2 mb-3">
          <input className="border p-2 flex-1" placeholder="Name" value={form.name} onChange={(e)=>setForm({...form, name: e.target.value})}/>
          <input className="border p-2 flex-1" placeholder="Email" value={form.email} onChange={(e)=>setForm({...form, email: e.target.value})}/>
          <input className="border p-2 flex-1" placeholder="City" value={form.city} onChange={(e)=>setForm({...form, city: e.target.value})}/>
        </div>
        <div>
          {editing ? (
            <>
              <button className="bg-green-600 text-white px-4 py-2 mr-2" onClick={()=>updateCustomer(editing)}>Save</button>
              <button className="bg-gray-300 px-4 py-2" onClick={()=>{setEditing(null); setForm({name:'',email:'',city:''})}}>Cancel</button>
            </>
          ) : (
            <button className="bg-blue-600 text-white px-4 py-2" onClick={createCustomer}>Add</button>
          )}
        </div>
      </div>

      <h2 className="text-xl mb-3">Customers</h2>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-gray-100">
              <th className="border px-3 py-2 text-left">ID</th>
              <th className="border px-3 py-2 text-left">Name</th>
              <th className="border px-3 py-2 text-left">Email</th>
              <th className="border px-3 py-2 text-left">City</th>
              <th className="border px-3 py-2 text-left">Actions</th>
            </tr>
          </thead>
          <tbody>
            {customers.map(c => (
              <tr key={c.id || Math.random()}>
                <td className="border px-3 py-2">{c.id}</td>
                <td className="border px-3 py-2">{c.name}</td>
                <td className="border px-3 py-2">{c.email}</td>
                <td className="border px-3 py-2">{c.city}</td>
                <td className="border px-3 py-2">
                  <button className="mr-2 px-3 py-1 bg-yellow-400" onClick={()=>{setEditing(c.id); setForm({name:c.name,email:c.email,city:c.city})}}>Edit</button>
                  <button className="px-3 py-1 bg-red-500 text-white" onClick={()=>deleteCustomer(c.id)}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
