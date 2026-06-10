from audiobench.daemon.factory import get_daemon_client
client = get_daemon_client()
results = client.search("happiness cannot be pursued", top_k=5)
for r in results:
    print(r)
