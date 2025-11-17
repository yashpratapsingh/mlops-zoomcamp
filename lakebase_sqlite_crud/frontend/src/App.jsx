import { useEffect, useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [items, setItems] = useState([])
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  const load = async () => {
    try {
      const res = await axios.get(`${API}/items/`)
      setItems(res.data)
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => { load() }, [])

  const add = async () => {
    await axios.post(`${API}/items/`, { name, description })
    setName(''); setDescription('')
    load()
  }

  const remove = async (id) => {
    await axios.delete(`${API}/items/${id}`)
    load()
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Lakebase CRUD (SQLite)</h1>

      <div className="mb-4">
        <input className="border p-2 mr-2" placeholder="Name" value={name} onChange={e=>setName(e.target.value)} />
        <input className="border p-2 mr-2" placeholder="Description" value={description} onChange={e=>setDescription(e.target.value)} />
        <button className="bg-blue-600 text-white px-4 py-2" onClick={add}>Add</button>
      </div>

      <div className="space-y-2">
        {items.map(it => (
          <div key={it.id} className="p-3 border rounded flex justify-between items-center">
            <div><strong>{it.name}</strong><div className="text-sm text-gray-600">{it.description}</div></div>
            <button className="bg-red-500 text-white px-3 py-1" onClick={()=>remove(it.id)}>Delete</button>
          </div>
        ))}
      </div>
    </div>
  )
}
