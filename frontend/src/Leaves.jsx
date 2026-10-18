export function LeafMark({ className = '' }) {
  return <svg className={className} viewBox="0 0 50 60" aria-hidden="true"><path d="M10 47C12 22 27 13 42 5c-2 24-6 38-26 41" fill="#33875b" stroke="#216445"/><path d="m6 55 31-43M18 38l-1-11m7 2 10-3" fill="none" stroke="#165d40" strokeWidth="2.5" strokeLinecap="round"/><path d="M17 36 34 13 24 33Z" fill="#7cbe90"/></svg>;
}

export function LeafIllustration({ decorative = false }) {
  if (decorative) return <svg className="footer-leaves" viewBox="0 0 390 110" preserveAspectRatio="none" aria-hidden="true">
    <g fill="#dceee2" stroke="#c5e2d0" strokeWidth="1.5">
      <path d="M0 32c53 5 69 27 73 78C32 100 10 72 0 32Z"/>
      <path d="M247 110C272 48 321 22 389 7c-10 62-45 93-111 103Z"/>
      <path d="M82 110c-8-9-12-17-7-29 16-2 26 5 26 14l-8 15Z" fill="#e1f0e6" stroke="none"/>
    </g>
    <g fill="none" stroke="#c6e3d0" strokeWidth="1.5"><path d="M0 32 73 110M15 49l3-14m12 31 4-21m9 38 5-23m9 38-21-3m-8-18-16-3m245 36 132-103M285 87l-1-29m18 15 2-30m17 18 2-28m17 15 1-25m-58 66 32 6m-13-20 30 4m-9-19 29 4m-11-16 28 2"/></g>
  </svg>;
  return <svg className={decorative ? 'footer-leaves' : 'leaf-illustration'} viewBox="0 0 330 210" aria-hidden="true">
    <defs><linearGradient id={decorative ? 'leaf-f' : 'leaf-h'} x2="0.6" y2="1"><stop stopColor="#b4d8bb"/><stop offset="1" stopColor="#62b382"/></linearGradient></defs>
    <circle cx="91" cy="77" r="58" fill="#edf7f0"/><ellipse cx="253" cy="101" rx="65" ry="47" fill="#f0f7f2"/>
    <g fill={`url(#${decorative ? 'leaf-f' : 'leaf-h'})`} stroke="#6cae80" strokeWidth="1">
      <path d="M152 199C120 171 77 164 68 52c60 33 82 58 84 147Z"/>
      <path d="M168 142C175 65 209 33 253 12c3 77-15 108-85 130Z"/>
      <path d="M152 199c33-51 68-58 120-58-37 41-71 52-120 58Z"/>
    </g>
    <g fill="none" stroke="#69aa7c" strokeWidth="1.5">
      <path d="m68 52 84 147 101-187M155 196l111-53M92 93l6-22m7 44 10-22m1 42 9-19m5 39 7-18M86 87l-11-1m26 25-21-4m37 30-28-9m43 30-29-8M187 102l-9-17m21-6-8-19m23-3-7-19m18 17 25-1m-38 24 31-3m-47 27 26-3M186 176l-3-14m20 6 2-14m17 5 4-12m-31 27 16 5m9-18 21 5"/>
    </g>
    <path d="m152 199 51-119m-51 119-33-72m33 72 82-43" fill="none" stroke="#62a773" strokeWidth="3"/>
    <g fill="#dceee3" stroke="#cae3d3" opacity=".8"><path d="M88 197C45 182 36 161 26 139c30 4 50 25 62 58Z"/><path d="M220 199c44-5 66-8 86-38-40 1-65 14-86 38Z"/><path d="M221 158c6-34 30-53 76-71-4 38-27 59-76 71Z"/></g>
    <g fill="none" stroke="#c6e2d0"><path d="m26 139 62 58m-40-36-2-12m15 25 1-13m244-1-86 38m40-17 8 6m-19-1 10 8m38-107-76 71m46-43 2-13m-20 30 1-16"/></g>
  </svg>;
}
