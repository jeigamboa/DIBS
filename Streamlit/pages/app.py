import streamlit as st
import overpy
import pandas as pd
import folium
from streamlit_folium import folium_static
import xml.etree.ElementTree as ET
<<<<<<< HEAD
#Test change
=======

>>>>>>> 1bde83dbbeac880f79556383e4238b5dd668bf31
# Title
st.title("BPI Branch Node Map - Katipunan Loyola (Quezon City)")

# Load and parse the embedded XML file
XML_PATH = "BPI.xml"  # File must exist in the same directory
tree = ET.parse(XML_PATH)
root = tree.getroot()

# Extract node references from the <way> element in order
way = root.find(".//way")
node_refs = [nd.attrib["ref"] for nd in way.findall("nd")]

# Extract metadata tags
tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
st.subheader("Branch Metadata")
st.json(tags)

# Use Overpass API to fetch node coordinates
api = overpy.Overpass()
query = f"""
(
  {"".join([f'node({ref});' for ref in node_refs])}
);
out body;
"""
result = api.query(query)

# Map node_id to (lat, lon)
node_coord_map = {str(node.id): (float(node.lat), float(node.lon)) for node in result.nodes}

# Reorder coordinates based on original node_refs
ordered_coords = [node_coord_map[ref] for ref in node_refs if ref in node_coord_map]

# Build DataFrame for display
df = pd.DataFrame([
    {"node_id": ref, "lat": lat, "lon": lon}
    for ref, (lat, lon) in zip(node_refs, ordered_coords)
])
st.subheader("Ordered Node Coordinates")
st.dataframe(df)

# Plot properly ordered polygon
st.subheader("Map Visualization")
center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(location=center, zoom_start=18)

folium.Polygon(
    locations=ordered_coords,
    color="blue",
    weight=2,
    fill=True,
    fill_color="blue",
    fill_opacity=0.4,
    tooltip=tags.get("name", "BPI")
).add_to(m)

folium_static(m)
