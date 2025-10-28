def create_kpi_card(title, value, delta=None):
    """Crée une carte KPI stylisée"""
    return f"""
    <div class="card">
        <h3>{title}</h3>
        <h1>{value}</h1>
        {f'<span class{"delta-positive" if delta >= 0 else "delta-negative"}>{delta:+.1f}%</span>' if delta else ''}
    </div>
    """