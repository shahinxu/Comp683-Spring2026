import matplotlib.pyplot as plt

# ==========================================
# 1. Data for Varying Dimensions
# ==========================================
dimensions_list = [16, 32, 64, 128, 256]
nmi_dims = [0.704, 0.751, 0.803, 0.821, 0.819]

plt.figure(figsize=(6, 4))
plt.plot(dimensions_list, nmi_dims, marker='o', color='b', linewidth=2)
plt.title('NMI vs. Number of Dimensions (Node2Vec)')
plt.xlabel('Dimensions')
plt.ylabel('Normalized Mutual Information (NMI)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('nmi_vs_dims.png', dpi=300, bbox_inches='tight')
print("Successfully generated: nmi_vs_dims.png")

# ==========================================
# 2. Data for Varying parameter 'q'
# ==========================================
q_values = [0.25, 0.5, 1.0, 2.0, 4.0]
nmi_qs = [0.762, 0.785, 0.821, 0.834, 0.811]

plt.figure(figsize=(6, 4))
plt.plot(q_values, nmi_qs, marker='s', color='r', linewidth=2)
plt.title('NMI vs. Parameter q (In-out parameter)')
plt.xlabel('q value')
plt.ylabel('Normalized Mutual Information (NMI)')
plt.xscale('log')
plt.xticks(q_values, labels=[str(q) for q in q_values])
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('nmi_vs_q.png', dpi=300, bbox_inches='tight')
print("Successfully generated: nmi_vs_q.png")