#!/usr/bin/env python
# coding: utf-8

# # Introduction to Hydrogeochemical Data Visualization with WQChartPy
# 
# ## Learning Objectives
# By the end of this notebook, you will be able to:
# 1. Understand the basics of hydrogeochemical data visualization
# 2. Install and import WQChartPy
# 3. Load and prepare water chemistry data for plotting
# 4. Create basic hydrogeochemical diagrams
# 5. Interpret common hydrogeochemical plots
# 
# ## What is WQChartPy?
# 
# WQChartPy is a Python package for creating geochemical diagrams used in water quality analysis and hydrogeochemistry. It can generate 12 different types of diagrams including:
# 
# - **Piper diagrams** (Triangle, Rectangle, Color-coded, Contour-filled)
# - **Stiff diagrams**
# - **Durov diagrams**
# - **Schoeller diagrams**
# - **Gibbs diagrams**
# - **Chadha diagrams**
# - **Gaillardet diagrams**
# - **HFE-D diagrams**
# - **Chernoff faces**

# ## Installation and Setup

# In[ ]:


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## Understanding Water Chemistry Data Format
# 
# WQChartPy requires data in a specific format. Let's examine the required columns:

# In[ ]:


from wqchartpy import triangle_piper

# Load the template dataset
df = pd.read_csv('./datasets/template_dataset.csv')

# Display the data
print("\nData shape:", df.shape)
df


# ### Required Columns Explained:
# 
# 1. **Sample**: Sample name or ID
# 2. **Label**: Group identifier (used for categorizing samples)
# 3. **Color**: Color for plotting (matplotlib color names or hex codes)
# 4. **Marker**: Marker style for plotting (matplotlib marker codes)
# 5. **Size**: Marker size for plotting
# 6. **Alpha**: Transparency (0 = transparent, 1 = opaque)
# 7. **pH**: pH value
# 8. **Major Ions**: Ca, Mg, Na, K, HCO3, CO3, Cl, SO4 (all in mg/L)
# 9. **TDS**: Total Dissolved Solids (mg/L)

# In[ ]:


# Basic statistics for the template data
print("Basic Statistics for Template Dataset:")
print("======================================")

# Select only the chemical parameter columns
chem_cols = ['pH', 'Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4', 'TDS']
df[chem_cols].describe().round(2)


# In[ ]:


print("\nSample Groups:")
df['Label'].value_counts()


# In[ ]:


print("\nTDS Range by Group:")
for group in df['Label'].unique():
    group_data = df[df['Label'] == group]
    print(f"{group}: {group_data['TDS'].min():.1f} - {group_data['TDS'].max():.1f} mg/L")


# ## Your First Hydrogeochemical Plot: Piper Diagram
# 
# The Piper diagram is one of the most widely used plots in hydrogeochemistry. It shows the relative concentrations of major cations and anions.

# In[ ]:


# Import the triangle Piper module
from wqchartpy import triangle_piper

triangle_piper.plot(
    df, unit='mg/L', 
    figname='./plots/example_triangle_piper', figformat='png'
)


# ## Understanding the Piper Diagram
# 
# The Piper diagram consists of:
# - **Left triangle**: Cations (Ca²⁺, Mg²⁺, Na⁺+K⁺)
# - **Right triangle**: Anions (Cl⁻, SO₄²⁻, HCO₃⁻+CO₃²⁻)
# - **Central diamond**: Combined cation and anion composition
# 
# ### Interpretation:
# - Points closer to corners indicate dominance of that ion
# - Points in the center indicate mixed water types
# - Clustering indicates similar water chemistry

# ## Creating Multiple Plot Types
# 
# Let's create several different plot types to compare their strengths:

# In[ ]:


# Stiff diagram - creates unique 'fingerprints' for each water sample
from wqchartpy import stiff

stiff.plot(
    df,  unit='mg/L', 
    figname='./plots/stiff_diagram_example', figformat='png'
)


# In[ ]:


# Gibbs diagram - helps identify water-rock interaction processes
from wqchartpy import gibbs

gibbs.plot(
    df, unit='mg/L', 
    figname='./plots/gibbs_example', figformat='png'
)


# In[ ]:


# Schoeller diagram - shows ion concentrations on a log scale
# Note: https://github.com/jyangfsu/WQChartPy/blob/main/wqchartpy/schoeller.py#L56-L76
from wqchartpy import schoeller

schoeller.plot(
    df, unit='mg/L', 
    figname='./plots/schoeller', figformat='png'
)


# ## Data Summary and Statistics

# ## Exercise: Create Your Own Plot
# 
# Try modifying the data and creating a new plot:

# In[ ]:


# Exercise: Modify colors and create a new Piper diagram
df_modified = df.copy()

# Change colors for each cluster - how to do this??
# TODO Need to do something here!

# Create the plot with modified colors
triangle_piper.plot(df_modified, 
                   unit='mg/L', 
                   figname='./plots/modified_piper', 
                   figformat='png')

print("Modified Piper diagram created with new colors!")


# ## Key Takeaways
# 
# 1. **Data Format**: WQChartPy requires specific column names and format
# 2. **Multiple Plot Types**: Different plots reveal different aspects of water chemistry
# 3. **Customization**: Colors, markers, and sizes can be customized for better visualization
# 4. **Interpretation**: Each plot type has specific interpretation guidelines
