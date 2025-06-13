from flask import Flask, jsonify, render_template,request, send_from_directory
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
from io import StringIO
import os
import base64
import plotly.graph_objects as go
import io  # Añadimos esta importación
import base64

# ... (código existente de app.py)

app = Flask(__name__)

os.makedirs('static', exist_ok=True)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Configuración para Render
# Obtener el puerto desde la variable de entorno PORT que proporciona Render
port = int(os.environ.get("PORT", 5000))

# Cargar y procesar datos
def load_data():
    df = pd.read_csv('data/sales_data_sample.csv', encoding='unicode_escape')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    
    # Procesamiento como en el notebook original
    df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 
               'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 
               'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER', 'STATUS']
    df = df.drop(df_drop, axis=1)
    
    # Convertir variables categóricas
    countries = pd.get_dummies(df['COUNTRY'], prefix='COUNTRY')
    productlines = pd.get_dummies(df['PRODUCTLINE'], prefix='PRODUCTLINE')
    dealsizes = pd.get_dummies(df['DEALSIZE'], prefix='DEALSIZE')
    
    df = pd.concat([df, countries, productlines, dealsizes], axis=1)
    df.drop(['COUNTRY', 'PRODUCTLINE', 'DEALSIZE', 'ORDERDATE'], axis=1, inplace=True)
    
    df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes
    
    return df

# Endpoint principal
@app.route('/')
def index():
    return render_template('index.html')

# API Endpoints
@app.route('/api/data/head')
def data_head():
    df = load_data()
    return jsonify(df.head(10).to_dict(orient='records'))

@app.route('/api/task1')
def task1():
    # Datos conceptuales para el diagrama
    diagram_data = {
        "nodes": [
            {"id": "problem", "label": "Problema", "level": 0},
            {"id": "data", "label": "Recolección\nde Datos", "level": 1},
            {"id": "analysis", "label": "Análisis", "level": 1},
            {"id": "segmentation", "label": "Segmentación", "level": 2},
            {"id": "strategy", "label": "Estrategia\nMarketing", "level": 3}
        ],
        "links": [
            {"source": "problem", "target": "data"},
            {"source": "data", "target": "analysis"},
            {"source": "analysis", "target": "segmentation"},
            {"source": "segmentation", "target": "strategy"}
        ]
    }
    return jsonify(diagram_data)

@app.route('/api/task2')
def task2():
    df = load_data()
    
    # Tipos de datos
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ['type', 'count']
    
    # Valores nulos (usamos datos originales para esto)
    df_raw = pd.read_csv('data/sales_data_sample.csv', encoding='unicode_escape')
    nulls = df_raw.isnull().sum().reset_index()
    nulls.columns = ['column', 'null_count']
    
    return jsonify({
        "dtypes": dtype_counts.to_dict(orient='records'),
        "nulls": nulls.to_dict(orient='records'),
        "data_info": {
            "original_columns": len(df_raw.columns),
            "final_columns": len(df.columns),
            "rows": len(df)
        }
    })

@app.route('/api/task3')
def task3():
    df_raw = pd.read_csv('data/sales_data_sample.csv', encoding='unicode_escape')
    
    country_counts = df_raw['COUNTRY'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    productline_counts = df_raw['PRODUCTLINE'].value_counts().reset_index()
    productline_counts.columns = ['productline', 'count']
    
    dealsize_counts = df_raw['DEALSIZE'].value_counts().reset_index()
    dealsize_counts.columns = ['dealsize', 'count']
    
    return jsonify({
        "country": country_counts.to_dict(orient='records'),
        "productline": productline_counts.to_dict(orient='records'),
        "dealsize": dealsize_counts.to_dict(orient='records')
    })

@app.route('/api/task4')
def task4():
    df = load_data()
    
    # Matriz de correlación (existente)
    corr_matrix = df.iloc[:, :10].corr().reset_index()
    corr_data = []
    for _, row in corr_matrix.iterrows():
        corr_data.append(row.to_dict())
    
    # Datos para pairplot (existente)
    sample_df = df.sample(n=100)
    pairplot_data = sample_df[['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'MONTH_ID']].to_dict(orient='records')
    
    # Distribuciones (existentes)
    sales_dist = df['SALES'].describe().to_dict()
    qty_dist = df['QUANTITYORDERED'].describe().to_dict()
    
    # Nuevos datos para las gráficas faltantes
    # 1. Gráfica de línea de ventas
    sales_df_group = df.groupby('MONTH_ID')['SALES'].sum().reset_index()
    sales_trend = {
        'months': sales_df_group['MONTH_ID'].tolist(),
        'sales': sales_df_group['SALES'].tolist()
    }
    
    # 2. Datos para distplots (excluyendo ORDERLINENUMBER)
    distplot_data = {}
    for i in range(8):
        col_name = df.columns[i]
        if col_name != 'ORDERLINENUMBER':
            distplot_data[col_name] = df[col_name].apply(float).tolist()
    
    return jsonify({
        "correlation": corr_data,
        "pairplot": pairplot_data,
        "sales_dist": sales_dist,
        "qty_dist": qty_dist,
        "sales_trend": sales_trend,  # Nueva
        "distplots": distplot_data   # Nueva
    })

@app.route('/api/task5')
def task5():
    # Datos conceptuales para K-Means
    points = [
        {"x": 1, "y": 2, "cluster": 0},
        {"x": 1.5, "y": 2.5, "cluster": 0},
        {"x": 3, "y": 3, "cluster": 0},
        {"x": 5, "y": 3, "cluster": 2},
        {"x": 3.5, "y": 2.5, "cluster": 0},
        {"x": 4.5, "y": 2.5, "cluster": 1},
        {"x": 3.5, "y": 4, "cluster": 1},
        {"x": 4.5, "y": 4, "cluster": 1},
        {"x": 5.5, "y": 4, "cluster": 2},
        {"x": 6, "y": 3.5, "cluster": 2}
    ]
    
    centroids = [
        {"x": 2, "y": 2.5, "cluster": 0},
        {"x": 4.5, "y": 3.5, "cluster": 1},
        {"x": 5.5, "y": 3, "cluster": 2}
    ]
    
    return jsonify({
        "points": points,
        "centroids": centroids,
        "description": {
            "what_is": "K-Means es un algoritmo de aprendizaje no supervisado que agrupa datos en K clusters basándose en su similitud.",
            "steps": [
                "Seleccionar número de clusters (K)",
                "Inicializar centroides aleatoriamente",
                "Asignar puntos al centroide más cercano",
                "Recalcular centroides como promedios",
                "Repetir hasta convergencia"
            ],
            "pros": [
                "Simple y rápido",
                "Escala bien con grandes datasets",
                "Fácil de interpretar"
            ],
            "cons": [
                "Requiere especificar K",
                "Sensible a valores atípicos",
                "Asume clusters esféricos"
            ]
        }
    })

@app.route('/api/task6')
def task6():
    df = load_data()
    scaler = StandardScaler()
    sales_df_scaled = scaler.fit_transform(df)
    
    scores = []
    range_values = list(range(1, 15))
    
    for i in range_values:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(sales_df_scaled)
        scores.append(kmeans.inertia_)
    
    return jsonify({
        "range": range_values,
        "scores": scores,
        "optimal_k": 5  # Determinado por el método del codo
    })

@app.route('/api/task7')
def task7():
    """
    Endpoint para el análisis de clusters con K-Means
    Devuelve:
    - Datos para visualización 2D
    - Estadísticas de clusters
    - Centroides
    - Histogramas por cluster
    """
    # Cargar y preparar datos
    df = load_data()
    scaler = StandardScaler()
    sales_df_scaled = scaler.fit_transform(df)
    
    # Aplicar K-Means (5 clusters)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sales_df_scaled)
    
    # PCA para visualización 2D
    pca = PCA(n_components=2)
    principal_comp = pca.fit_transform(sales_df_scaled)
    pca_df = pd.DataFrame(data=principal_comp, columns=['x', 'y'])
    pca_df['cluster'] = labels
    
    # Estadísticas por cluster
    sale_df_cluster = pd.concat([df, pd.DataFrame({'cluster':labels})], axis=1)
    cluster_stats = []
    for i in range(5):
        cluster_data = sale_df_cluster[sale_df_cluster['cluster'] == i]
        stats = {
            "cluster": i,
            "size": len(cluster_data),
            "avg_quantity": cluster_data['QUANTITYORDERED'].mean(),
            "avg_price": cluster_data['PRICEEACH'].mean(),
            "total_sales": cluster_data['SALES'].sum()
        }
        cluster_stats.append(stats)
    
    # Centroides (transformados inversamente para interpretación)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers = pd.DataFrame(data=cluster_centers, columns=df.columns)
    
    # Datos para histogramas (primeras 8 columnas)
    histograms = {}
    for col in df.columns[:8]:
        histograms[col] = []
        for cluster_num in range(5):
            cluster_data = sale_df_cluster[sale_df_cluster['cluster'] == cluster_num][col]
            histograms[col].append(cluster_data.tolist())
    
    return jsonify({
        "clusters": pca_df.to_dict(orient='records'),  # Para visualización 2D
        "cluster_stats": cluster_stats,  # Estadísticas
        "centroids": cluster_centers.to_dict(orient='records'),  # Centroides
        "histograms": histograms,  # Datos para histogramas
        "feature_names": df.columns[:8].tolist(),  # Nombres de columnas
        "cluster_descriptions": [  # Descripciones interpretativas
            "Clientes que compran en grandes cantidades y gastan mucho",
            "Clientes que prefieren productos premium",
            "Clientes ocasionales con bajo gasto",
            "Clientes estacionales con gasto moderado",
            "Clientes regulares con gasto consistente"
        ]
    })

@app.route('/api/task8')
def task8():
    df = load_data()
    scaler = StandardScaler()
    sales_df_scaled = scaler.fit_transform(df)
    
    # PCA 3D
    pca = PCA(n_components=3)
    principal_comp = pca.fit_transform(sales_df_scaled)
    pca_df = pd.DataFrame(data=principal_comp, columns=['x', 'y', 'z'])
    
    # K-Means para clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    pca_df['cluster'] = kmeans.fit_predict(sales_df_scaled)
    
    # Varianza explicada
    pca_full = PCA().fit(sales_df_scaled)
    variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    return jsonify({
        "points": pca_df.to_dict(orient='records'),
        "variance": variance.tolist(),
        "components": pca_full.components_.tolist()
    })

@app.route('/api/task9')
def task9():
    # Datos conceptuales para Autoencoder
    layers = [
        {"id": "input", "name": "Input", "units": 37, "x": 0.1, "y": 0.5},
        {"id": "enc1", "name": "Encoder", "units": 50, "x": 0.3, "y": 0.5},
        {"id": "enc2", "name": "Encoder", "units": 500, "x": 0.5, "y": 0.5},
        {"id": "bottleneck", "name": "Bottleneck", "units": 8, "x": 0.7, "y": 0.5},
        {"id": "dec1", "name": "Decoder", "units": 500, "x": 0.9, "y": 0.5},
        {"id": "output", "name": "Output", "units": 37, "x": 1.1, "y": 0.5}
    ]
    
    connections = [
        {"source": "input", "target": "enc1"},
        {"source": "enc1", "target": "enc2"},
        {"source": "enc2", "target": "bottleneck"},
        {"source": "bottleneck", "target": "dec1"},
        {"source": "dec1", "target": "output"}
    ]
    
    return jsonify({
        "layers": layers,
        "connections": connections,
        "description": {
            "what_is": "Un autoencoder es una red neuronal que aprende a comprimir datos (codificar) y luego reconstruirlos (decodificar).",
            "components": [
                {"name": "Encoder", "desc": "Reduce la dimensionalidad de los datos"},
                {"name": "Bottleneck", "desc": "Representación comprimida de los datos"},
                {"name": "Decoder", "desc": "Reconstruye los datos desde la representación comprimida"}
            ],
            "pros": [
                "Reducción no lineal de dimensionalidad",
                "Capaz de aprender características complejas",
                "Útil para datos no etiquetados"
            ],
            "cons": [
                "Requiere más datos que métodos lineales como PCA",
                "Más difícil de interpretar",
                "Computacionalmente más costoso"
            ]
        }
    })
    



@app.route('/api/mri-report', methods=['POST'])
def mri_report():
    try:
        # Obtener datos de JSON, query params o form-data
        data = request.get_json(silent=True)
        if data is None:
            data = {
                'diagnosis': request.form.get('diagnosis', request.args.get('diagnosis', 'No diagnosis')),
                'confidence': request.form.get('confidence', request.args.get('confidence', '0'))
            }
        
        diagnosis = data.get('diagnosis', 'No diagnosis')
        confidence = data.get('confidence', 0)
        
        # Convertir confidence a float
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            print(f"[ERROR] Confidence inválido: {confidence}")
            return jsonify({'status': 'error', 'message': 'Confidence debe ser un número válido'}), 400
        
        print(f"[DEBUG] mri-report - Diagnosis: {diagnosis}, Confidence: {confidence}")
        
        # Crear gráfica de barras
        fig = go.Figure(data=[
            go.Bar(
                x=['Confianza'],
                y=[confidence],
                text=[f'{confidence:.2f}%'],
                textposition='auto',
                marker_color='#4CA1AF' if 'No se detectó' in diagnosis else '#D4F483'
            )
        ])
        fig.update_layout(
            title=f'Resultado del Análisis MRI: {diagnosis}',
            yaxis_title='Porcentaje',
            xaxis_title='',
            showlegend=False,
            height=300,
            width=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Generar imagen PNG
        buffer = io.BytesIO()
        fig.write_image(buffer, format='png', scale=1)
        buffer.seek(0)
        
        # Guardar imagen
        image_path = 'static/mri-report.png'
        with open(image_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        # Generar URL
        graph_html = f'<img src="/static/mri-report.png" alt="Gráfica MRI" style="max-width: 100%; height: auto;">'
        
        return jsonify({
            'status': 'success',
            'diagnosis': diagnosis,
            'confidence': confidence,
            'graph': graph_html
        })
    except Exception as e:
        print(f"[ERROR] mri_report: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/task9-visualization', methods=['GET'])
def task9_visualization():
    try:
        # Datos del autoencoder
        task9_data = {
            "connections": [
                {"source": "input", "target": "enc1"},
                {"source": "enc1", "target": "enc2"},
                {"source": "enc2", "target": "bottleneck"},
                {"source": "bottleneck", "target": "dec1"},
                {"source": "dec1", "target": "output"}
            ],
            "description": {
                "components": [
                    {"name": "Encoder", "desc": "Reduce la dimensionalidad de los datos"},
                    {"name": "Bottleneck", "desc": "Representación comprimida de los datos"},
                    {"name": "Decoder", "desc": "Reconstruye los datos desde la representación comprimida"}
                ],
                "pros": [
                    "Reducción no lineal de dimensionalidad",
                    "Capaz de aprender características complejas",
                    "Útil para datos no etiquetados"
                ],
                "cons": [
                    "Requiere más datos que métodos lineales como PCA",
                    "Más difícil de interpretar",
                    "Computacionalmente más costoso"
                ],
                "what_is": "Un autoencoder es una red neuronal que aprende a comprimir datos (codificar) y luego reconstruirlos (decodificar)."
            },
            "layers": [
                {"id": "input", "name": "Input", "units": 37, "x": 0.1, "y": 0.5},
                {"id": "enc1", "name": "Encoder", "units": 50, "x": 0.3, "y": 0.5},
                {"id": "enc2", "name": "Encoder", "units": 500, "x": 0.5, "y": 0.5},
                {"id": "bottleneck", "name": "Bottleneck", "units": 8, "x": 0.7, "y": 0.5},
                {"id": "dec1", "name": "Decoder", "units": 500, "x": 0.9, "y": 0.5},
                {"id": "output", "name": "Output", "units": 37, "x": 1.1, "y": 0.5}
            ]
        }
        
        print("[DEBUG] Generando gráfica de autoencoder...")
        
        # Configurar datos para la gráfica
        layers = task9_data["layers"]
        x_positions = [i for i in range(len(layers))]
        y_positions = [layer["units"] for layer in layers]
        labels = [f"{layer['name']} ({layer['units']})" for layer in layers]
        
        # Colores para las capas
        colors = ['#FF6B6B' if layer["name"] in ["Input", "Output"] else '#FFE66D' if layer["name"] == "Bottleneck" else '#4ECDC4' for layer in layers]
        
        # Crear gráfica
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_positions,
            y=y_positions,
            text=labels,
            textposition='auto',
            marker_color=colors,
            showlegend=False
        ))
        
        # Agregar conexiones
        for i in range(len(x_positions) - 1):
            fig.add_shape(
                type="line",
                x0=x_positions[i] + 0.4,
                y0=y_positions[i] / 2,
                x1=x_positions[i + 1] - 0.4,
                y1=y_positions[i + 1] / 2,
                line=dict(color="black", width=2)
            )
            fig.add_annotation(
                x=x_positions[i + 1] - 0.4,
                y=y_positions[i + 1] / 2,
                ax=x_positions[i] + 0.4,
                ay=y_positions[i] / 2,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                arrowhead=2,
                arrowwidth=2,
                arrowcolor="black",
                showarrow=True
            )
        
        fig.update_layout(
            title='Arquitectura del Autoencoder',
            xaxis=dict(
                tickmode='array',
                tickvals=x_positions,
                ticktext=[layer["name"] for layer in layers],
                title="Capas"
            ),
            yaxis=dict(title="Unidades", range=[0, max(y_positions) * 1.1]),
            height=400,
            width=600,
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Generar imagen PNG
        buffer = io.BytesIO()
        fig.write_image(buffer, format='png', scale=1)
        buffer.seek(0)
        
        # Guardar imagen
        image_path = 'static/autoencoder.png'
        with open(image_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        # Generar URL
        graph_html = f'<img src="/static/autoencoder.png" alt="Arquitectura del Autoencoder" style="max-width: 100%; height: auto;">'
        
        # Generar tabla HTML
        table_html = """
        <table style="width: 100%; border: 1px solid #ddd; font-family: Arial, sans-serif;">
            <thead>
                <tr style="background-color: #f4f4f4;">
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Componente</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Descripción</th>
                </tr>
            </thead>
            <tbody>
                {components}
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Ventajas</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{pros}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Desventajas</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{cons}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">¿Qué es?</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{what_is}</td>
                </tr>
            </tbody>
        </table>
        """
        components_rows = "".join([
            f'<tr><td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{comp["name"]}</td><td style="padding: 10px; border: 1px solid #ddd;">{comp["desc"]}</td></tr>'
            for comp in task9_data["description"]["components"]
        ])
        pros = "<br>• ".join([""] + task9_data["description"]["pros"])
        cons = "<br>• ".join([""] + task9_data["description"]["cons"])
        what_is = task9_data["description"]["what_is"]
        table_html = table_html.format(components=components_rows, pros=pros, cons=cons, what_is=what_is)
        
        # Generar tabla plain-text
        plain_table = (
            "Componente | Descripción\n"
            "-----------|------------\n" +
            "\n".join([f"{comp['name']} | {comp['desc']}" for comp in task9_data["description"]["components"]]) +
            f"\n-----------|------------\n"
            f"Ventajas | {' • '.join(task9_data['description']['pros'])}\n"
            f"Desventajas | {' • '.join(task9_data['description']['cons'])}\n"
            f"¿Qué es? | {task9_data['description']['what_is']}"
        )
        
        return jsonify({
            'status': 'success',
            'data': task9_data,
            'graph': graph_html,
            'table': table_html,
            'plain_table': plain_table,
            'debug_info': {
                'image_size': os.path.getsize(image_path),
                'has_image': os.path.exists(image_path)
            }
        })
    except Exception as e:
        print(f"[ERROR] task9_visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'graph': '<p style="color: red;">Error generando gráfica del autoencoder</p>',
            'table': table_html,
            'plain_table': plain_table
        }), 500

@app.route('/api/task10')
def task10():
    df = load_data()
    scaler = StandardScaler()
    sales_df_scaled = scaler.fit_transform(df)
    
    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=8)
    reduced_data = pca.fit_transform(sales_df_scaled)
    
    # Gráfica de elbow method
    scores = []
    range_values = range(1, 15)
    for i in range_values:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(reduced_data)
        scores.append(kmeans.inertia_)
    
    # Clustering con 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(reduced_data)
    
    # PCA para visualización 3D
    pca_3d = PCA(n_components=3)
    principal_comp = pca_3d.fit_transform(reduced_data)
    pca_df = pd.DataFrame(data=principal_comp, columns=['x', 'y', 'z'])
    pca_df['cluster'] = labels
    
    # Preparar datos para histogramas (solo una vez)
    df_cluster = pd.concat([df, pd.DataFrame({'cluster': labels})], axis=1)
    histograms = {}
    for col in df.columns[:8]:  # Primeras 8 columnas
        histograms[col] = []
        for cluster_num in range(3):  # 3 clusters
            cluster_data = df_cluster[df_cluster['cluster'] == cluster_num][col]
            histograms[col].append({
                'values': cluster_data.tolist(),
                'mean': float(cluster_data.mean()),
                'std': float(cluster_data.std()),
                'min': float(cluster_data.min()),
                'max': float(cluster_data.max())
            })
    
    # Estadísticas por cluster
    cluster_stats = []
    for i in range(3):
        cluster_data = df_cluster[df_cluster['cluster'] == i]
        stats = {
            'size': len(cluster_data),
            'avg_quantity': float(cluster_data['QUANTITYORDERED'].mean()),
            'avg_price': float(cluster_data['PRICEEACH'].mean()),
            'avg_sales': float(cluster_data['SALES'].mean())
        }
        cluster_stats.append(stats)
    
    return jsonify({
        "points": pca_df.to_dict(orient='records'),
        "histograms": histograms,
        "cluster_stats": cluster_stats,
        "feature_names": df.columns[:8].tolist(),
        "elbow_data": {
            "clusters": list(range_values),
            "scores": scores
        },
        "cluster_descriptions": [
            "Clientes que compran en grandes cantidades (media: {:.1f} unidades) y prefieren productos caros (${:.2f} promedio)".format(
                cluster_stats[0]['avg_quantity'], cluster_stats[0]['avg_price']),
            "Clientes con compras promedio (media: {:.1f} unidades) que prefieren productos de alto precio (${:.2f} promedio)".format(
                cluster_stats[1]['avg_quantity'], cluster_stats[1]['avg_price']),
            "Clientes que compran en pequeñas cantidades (media: {:.1f} unidades) y prefieren productos económicos (${:.2f} promedio)".format(
                cluster_stats[2]['avg_quantity'], cluster_stats[2]['avg_price'])
        ],
        "cluster_colors": ["#4CA1AF", "#2C3E50", "#D4B483"]
    })
    
    

if __name__ == '__main__':
    # Iniciar la aplicación Flask en el puerto que provee Render y en 0.0.0.0
    app.run(host='0.0.0.0', port=port, debug=False)
