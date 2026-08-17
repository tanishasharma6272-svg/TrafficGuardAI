import { useEffect, useState } from 'react'

export function useLiveClock() {
  const [clock, setClock] = useState(new Date())

  useEffect(() => {
    const tick = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(tick)
  }, [])

  return clock
}
