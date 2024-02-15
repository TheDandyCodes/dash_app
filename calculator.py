import numpy as np
import pandas as pd
import plotly.graph_objects as go


liq_spread_table = pd.read_csv('pricing_parameters/LS.csv')
fix_params = pd.read_csv('pricing_parameters/params.csv')


def real_balance_calc(tasa_interes_anual, num_pagos, monto_prestamo, cancelacion=0):

    if tasa_interes_anual==0.0:
        tasa_interes_anual = 0.1**7
    i = tasa_interes_anual/12
    nr = num_pagos - cancelacion
    ip = monto_prestamo
    mp = (ip * i) / (1 - (1 + i)**-num_pagos)
    A = ((1+i)**nr -1)
    B = (ip-(mp/i))
    C = A*B/nr
    return i**-1*(C+mp)

def duration_2(n_pagos, monto, tin, t):
    if tin==0.0:
        tin = 0.1**10
    npv = (monto * tin/12) / (1 - (1 + tin/12)**-n_pagos) / ((1+tin/12)**t)
    return np.sum(t*npv) / monto

def liquidity_spread_calc(liq_spread_table, duration):

    if len(liq_spread_table[(liq_spread_table.Time<=duration)].values) == 0:
        prev_ls_term = liq_spread_table.iloc[0:].LS.values[0]
        next_ls_term = prev_ls_term
    else:
        prev_time_term = liq_spread_table[(liq_spread_table.Time<=duration)].Time[-1:].values[0]

        if prev_time_term == liq_spread_table.iloc[-1:].Time.values[0]:
            prev_ls_term = liq_spread_table.iloc[-1:].LS.values[0]
            next_ls_term = prev_ls_term

        else:
            next_time_term = liq_spread_table.iloc[liq_spread_table[(liq_spread_table.Time<=duration)].index + 1, :].Time[-1:].values[0]
            prev_ls_term = liq_spread_table[liq_spread_table.Time == prev_time_term].LS.values[0]
            next_ls_term = liq_spread_table[liq_spread_table.Time == next_time_term].LS.values[0]

    if next_ls_term != prev_ls_term:
        return ((duration - prev_time_term) * (next_ls_term-prev_ls_term)/(next_time_term-prev_time_term)) + prev_ls_term
    else:
        return next_ls_term

def frenchAmortizationCalculator(monto_prestamo, tasa_interes_anual=0, num_pagos=1, cancelacion=0):

    # params = {'Monto':monto_prestamo, 'TIN':tasa_interes_anual, 'Pagos':num_pagos, 'Cancelacion':cancelacion, 'Comision_pagada': comision_pagada, 'Comision_apertura':comision_apertura, 'ITR':ITR, 'other':other, 'op_exp':op_exp, 'EL':EL, 'RWA':RWA}

    if tasa_interes_anual == 0.0:
        tasa_interes_anual = 0.1**10

    tasa_interes_mensual = tasa_interes_anual/12
    #cuota_fija = (monto_prestamo*tasa_interes_mensual*(1+tasa_interes_mensual)**(plazo*pagos_anuales))/((1+tasa_interes_mensual)**(plazo*pagos_anuales)-1)
    cuota_fija = (monto_prestamo * tasa_interes_mensual) / (1 - (1 + tasa_interes_mensual)**-num_pagos)
    saldo_final = monto_prestamo

    amortizaciones = []

    if cancelacion > 0:
        for i in range(1, (num_pagos-cancelacion)+1):
            saldo_inicial = saldo_final
            interes = tasa_interes_mensual*saldo_inicial
            amortizacion = cuota_fija-interes
            saldo_inicial = saldo_final
            if i==num_pagos-cancelacion:
                saldo_final = 0
            else:
                saldo_final -= amortizacion
            row = np.round([i, saldo_inicial, interes, amortizacion, cuota_fija, saldo_final], 3)
            amortizaciones.append(row)
        t = np.arange(1, (num_pagos-cancelacion)+1)

    else:
        for i in range(1, num_pagos+1):
            saldo_inicial = saldo_final
            interes = tasa_interes_mensual*saldo_inicial
            amortizacion = cuota_fija-interes
            saldo_inicial = saldo_final
            saldo_final -= amortizacion
            row = np.round([i, saldo_inicial, interes, amortizacion, cuota_fija, saldo_final], 3)
            amortizaciones.append(row)
        t = np.arange(1, num_pagos+1)

    duration = duration_2(num_pagos, monto_prestamo, tasa_interes_anual, t)

    cuadro = pd.DataFrame(amortizaciones, columns=['Mes', 'Saldo inicial', 'Interes', 'Principal', 'Cuota Fija', 'Saldo Final'])
    
    # li = cuadro['Saldo Final'][:-1].tolist()
    # li.append(monto_prestamo)
    # real_balance = np.mean(li)

    real_balance = real_balance_calc(tasa_interes_anual, num_pagos, monto_prestamo, cancelacion)
    total = pd.DataFrame({'Interes':sum(cuadro['Interes']), 'Principal':sum(cuadro['Principal'])+(cuadro['Saldo inicial'][-1:] - cuadro.Principal[-1:]).values[0], 'Total cuotas':(sum(cuadro['Interes'])+sum(cuadro['Principal'])+(cuadro['Saldo inicial'][-1:] - cuadro.Principal[-1:]).values[0]), 'Saldo medio':real_balance, 'Duración':duration, 'Plazo real':num_pagos-cancelacion}, index=['Total'])
    

    return np.round(cuadro,2), np.round(total,4)

def getIndicators(monto_prestamo, tasa_interes_anual, num_pagos, cancelacion=0, comision_terceros=0, comision_rappels=0, comision_apertura=0, ITR=0, other=0, op_exp=0, EL=0, RWA=1):
    pl = {}
    pl['TIN'] = tasa_interes_anual
    t = np.arange(1, (num_pagos-cancelacion)+1)
    duration = duration_2(num_pagos, monto_prestamo, tasa_interes_anual, t)
    real_balance = real_balance_calc(tasa_interes_anual, num_pagos, monto_prestamo, cancelacion)
    fees_collected = comision_apertura*12*(real_balance*(num_pagos-cancelacion))**-1
    fees_terceros = comision_terceros*12*(real_balance*(num_pagos-cancelacion))**-1
    fees_rappels = comision_rappels*12*(real_balance*(num_pagos-cancelacion))**-1
    fees_paid = fees_terceros+fees_rappels
    financial_fees = fees_collected-fees_paid
    pl['financial_fees'] = financial_fees
    # other = other*12*(real_balance*(num_pagos-cancelacion))**-1
    # op_exp = op_exp*12*(real_balance*(num_pagos-cancelacion))**-1
    LS = liquidity_spread_calc(liq_spread_table, duration) / 10**4
    TII = tasa_interes_anual + financial_fees
    pl['TII'] = TII
    TIE = LS + ITR
    pl['TIE'] = TIE
    NII = TII - TIE
    pl['NII'] = NII
    pl['non_fin_fees'] = other
    GM = NII - other
    pl['GM'] = GM
    pl['op_exp'] = op_exp
    NOI = GM - op_exp
    pl['NOI'] = NOI
    PBT = NOI - EL
    pl['EL'] = EL
    pl['PBT'] = PBT
    PAT = 0.715*PBT
    pl['PAT'] = PAT
    RORWA = PAT*monto_prestamo / RWA
    pl['RWA'] = RWA
    pl['RORWA'] = RORWA
    
    cuenta = pd.DataFrame(pl, index=['RORWA']).T.reset_index()
    cuenta.columns = ['Valores', 'Contrato']
    # Añadir calculo TII, TIE y NII para hayar finalmente PBT y PAT, que es lo que se va a mostrar
    indicators = pd.DataFrame({'LS':LS, 'PBT':PBT, 'PAT':PAT, 'RORWA':RORWA}, index=['RORWA'])
    return np.round(indicators, 4)

def RORWA_flag(actual_rorwa):
    flag = 'Green'
    if actual_rorwa >= 2: flag = 'Green'
    elif actual_rorwa >= 1 and actual_rorwa<2: flag = 'Orange'
    else: flag = 'Red'
    return flag

def line_plot(table, x_var):
    fig = go.Figure()

    # Crear y personalizar las trazas
    fig.add_trace(go.Scatter(x=table[x_var], y=table.RORWA, name='ITR',
                            line=dict(color='rgb(102,124,165)', width=3),
                            mode='lines+markers',
                            marker=dict(size=8),
                            # line_shape='spline' 
                            ))

    # Editar el diseño
    fig.update_layout(title='Evolución',
                    xaxis_title=x_var+' %',
                    yaxis_title='RORWA %', 
                    height=500,
                    width=850,
                    plot_bgcolor='white',
                    #yaxis=dict(range=[-1.5, 4]),
                    xaxis=dict(showline=True, linecolor='rgb(184,184,184)', linewidth=2),
                    yaxis=dict(showline=True, linecolor='rgb(184,184,184)', linewidth=2),
                    )
    return fig

import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc

# Inicializa la aplicación Dash
app = dash.Dash(__name__)
server = app.server

filas_gris = [4, 6, 8]
filas_amarilla = [10, 11, 13]

opciones_dropdown = [
    {'label': 'Consumo', 'value': 'consumo'},
    {'label': 'Auto Usado', 'value': 'auto_usado'},
    {'label': 'Auto Nuevo', 'value': 'auto_nuevo'},
    {'label': 'Directo', 'value': 'directo'},
]

# Define el diseño del dashboard con 4 campos de entrada
app.layout = html.Div([
    html.H1("Calculadora Comercial", style={'color':'rgb(0,26,72)'}),
    html.H4("Introduzca el tipo de producto que se desea financiar", style={'color':'rgb(0,26,72)'}),
        dcc.Dropdown(
        id='contrato-input',
        value='consumo',
        options=opciones_dropdown, 
        style={'width': '200px', 'height': '30px'},

    ),
    
    html.H2("Tabla de inputs", style={'color':'rgb(0,26,72)'}),
    html.H4("Introduzca las características del préstamo", style={'color':'rgb(0,26,72)'}),
    html.Table([
        html.Tr([
            html.Td('Inversión:'),
            html.Td(dcc.Input(id='input-valor-1', type='number')),
        ]),
        html.Tr([
            html.Td('TIN (sobre 1):'),
            html.Td(dcc.Input(id='input-valor-2', type='number')),
        ]),
        html.Tr([
            html.Td('Pagos:'),
            html.Td(dcc.Input(id='input-valor-3', type='number')),
        ]),
        # html.Tr([
        #     html.Td('Cancelación:'),
        #     html.Td(dcc.Input(id='input-valor-4', type='number')),
        # ]),
        html.Tr([
            html.Td('Comisiones pagadas a terceros:'),
            html.Td(dcc.Input(id='input-valor-5', type='number')),
        ]),
        html.Tr([
            html.Td('Comisiones pagadas rappels:'),
            html.Td(dcc.Input(id='input-valor-6', type='number')),
        ]),
        html.Tr([
            html.Td('Comisiones de apertura:'),
            html.Td(dcc.Input(id='input-valor-7', type='number')),
        ]),
    ], style={'margin': '20px'}),
    
    html.H2("Parámetros por producto", style={'color':'rgb(0,26,72)'}),
    html.H4("Son fijos desagregando por lo que se quiera. (Esto no se vería)", style={'color':'rgb(0,26,72)'}),
    html.Div(dash_table.DataTable(id='tabla-params', style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'}), style={'margin-bottom': '20px'},),
    
    dmc.Container([
    dmc.Grid([
        dmc.Col([
            html.Div([
            html.H2("Valores resultantes totales", style={'color':'rgb(0,26,72)'}),
            dash_table.DataTable(id='tabla-1', style_cell={
        'height': 'auto',
        'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
        'whiteSpace': 'normal'
    }, style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'})])], span='auto'),
        dmc.Col([
            html.Div([
            html.H2("Beneficio de la operación", style={'color':'rgb(0,26,72)'}),
            dash_table.DataTable(id='tabla-3', style_data_conditional=[
                {
                    'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} >= 0.02'},
                    'backgroundColor': 'green',
                    'color': 'white',
                },
                {
                    'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 0.02 && {RORWA} >= 0.01'},
                    'backgroundColor': 'orange',
                },
                {
                    'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 0.01'},
                    'backgroundColor': 'red',
                    'color': 'white',
                },], style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'})])]
                , span='auto'),
    ]),

], fluid=True),

dmc.Container([
    dmc.Grid([
        dmc.Col([
            html.Div([
            html.H2("Cuadro de amortización francesa", style={'color':'rgb(0,26,72)'}),
            dash_table.DataTable(id='tabla-2', style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'})])]
                , span='auto'),
        dmc.Col([
            html.Div([
            html.H2("Cuadro de rentabilidad (P&L)", style={'color':'rgb(0,26,72)'}),
            dash_table.DataTable(id='tabla-4', style_data_conditional=[
                {
                    'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} >= 0.02'},
                    'backgroundColor': 'green',
                    'color': 'white',
                },
                {
                    'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 0.02 && {RORWA} >= 0.01'},
                    'backgroundColor': 'orange',
                },
                {
                    'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 0.01'},
                    'backgroundColor': 'red',
                    'color': 'white',
                },]+[
            {
                'if': {'row_index': i},
                'backgroundColor': 'rgb(178,178,178)',
                'color': 'white',
                'font-weight': 'bold',
            }
            for i in filas_gris
        ]+[
            {
                'if': {'row_index': i},
                'backgroundColor': 'rgb(193,168,51)',
                'color': 'white',
                'font-weight': 'bold',
            }
            for i in filas_amarilla
        ]
                , style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'})])], span='auto'),
    ]),

], fluid=True),

    # html.H2("Valores resultantes totales"),
    # # Gráfico o visualización
    # html.Div(id='text'),
    # html.Div(dash_table.DataTable(id='tabla-1', style_cell={
    #     'height': 'auto',
    #     'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
    #     'whiteSpace': 'normal'
    # }, style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'}), style={'margin-bottom': '20px'}),
    # html.H2("Beneficio de la operación"),
    # html.Div(dash_table.DataTable(id='tabla-3', style_data_conditional=[
    #             {
    #                 'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} >= 2'},
    #                 'backgroundColor': 'green',
    #                 'color': 'white',
    #             },
    #             {
    #                 'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 2 && {RORWA} >= 1'},
    #                 'backgroundColor': 'orange',
    #             },
    #             {
    #                 'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 1'},
    #                 'backgroundColor': 'red',
    #                 'color': 'white',
    #             },], style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'}), style={'margin-bottom': '20px'}),
    # html.H2("Cuadro de rentabilidad (P&L)"),
    # html.Div(dash_table.DataTable(id='tabla-4', style_data_conditional=[
    #             {
    #                 'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} >= 2'},
    #                 'backgroundColor': 'green',
    #                 'color': 'white',
    #             },
    #             {
    #                 'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 2 && {RORWA} >= 1'},
    #                 'backgroundColor': 'orange',
    #             },
    #             {
    #                 'if': {'column_id': 'RORWA', 'filter_query': '{RORWA} < 1'},
    #                 'backgroundColor': 'red',
    #                 'color': 'white',
    #             },]+[
    #         {
    #             'if': {'row_index': i},
    #             'backgroundColor': 'rgb(178,178,178)',
    #             'color': 'white',
    #             'font-weight': 'bold',
    #         }
    #         for i in filas_gris
    #     ]+[
    #         {
    #             'if': {'row_index': i},
    #             'backgroundColor': 'rgb(193,168,51)',
    #             'color': 'white',
    #             'font-weight': 'bold',
    #         }
    #         for i in filas_amarilla
    #     ]
    #             , style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'}), style={'margin-bottom': '20px'}),
    # html.H2("Cuadro de amortización francesa"),
    # html.Div(dash_table.DataTable(id='tabla-2', style_header={'backgroundColor': 'rgb(0,26,72)', 'fontWeight': 'bold', 'color': 'white'}), style={'margin-bottom': '20px'}),
    #html.Div(dash_table.DataTable(id='tabla-5'), style={'margin-bottom': '20px'}),

    html.Div([
        dcc.Graph(id='line-6', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='line-1', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='line-2', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='line-3', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='line-4', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='line-5', style={'width': '33%', 'display': 'inline-block'}),
    ]),

    

    # dcc.Graph(
    #     id='grafico-barras',
    #     figure=px.bar(df.groupby('Cluster').mean().reset_index(), x='Cluster', y='RORWA'),
    # ),

    # dcc.Graph(
    #     id='grafico-scatter',
    #     figure=px.scatter(df, x='NOI', y='PAT', color='Cluster'),
    # ),
    
    # dcc.Graph(
    #     id='grafico-scatter-2',
    #     figure=px.scatter(df, x='PAT', y='RORWA', color='Cluster'),
    # ),

    # dcc.Graph(
    #     id='grafico-scatter-3',
    #     figure=px.scatter_3d(df, x='TIN', y='Monto', z='Pagos', size='LS', color='Cluster'),
    # ),
], style={'font-family': 'sans-serif'},)

# Función de callback para crear un DataFrame y actualizar la visualización
@app.callback(
    [Output('tabla-1', 'data'), Output('tabla-2', 'data'), Output('tabla-3', 'data'), Output('line-1', 'figure'),  Output('line-2', 'figure'), Output('line-3', 'figure'), Output('tabla-4', 'data'), Output('line-4', 'figure'), Output('line-5', 'figure'), Output('line-6', 'figure'), Output('tabla-params', 'data')], #Output('line-2', 'figure'), Output('line-3', 'figure')
    [
        Input('input-valor-1', 'value'),
        Input('input-valor-2', 'value'),
        Input('input-valor-3', 'value'),
        #Input('input-valor-4', 'value'),
        Input('input-valor-5', 'value'),
        Input('input-valor-6', 'value'),
        Input('input-valor-7', 'value'),
        Input('contrato-input', 'value'),
    ]
)
def table(valor1, valor2, valor3, valor5, valor6, valor7, valor8):
    # Crea un DataFrame con los valores ingresados
    valor4 = fix_params[fix_params.contrato == valor8]['plazo cancelado'].values[0]
    tab, tot = frenchAmortizationCalculator(valor1, valor2, valor3, fix_params[fix_params.contrato == valor8]['plazo cancelado'].values[0])
    ind = getIndicators(valor1, valor2, valor3, valor4, valor5, valor6, valor7, fix_params[fix_params.contrato == valor8].ITR.values[0], fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0], fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0], fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0], fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1)
    
    #fix_inputs = pd.DataFrame({'Inputs':['Contrato', 'ITR', 'Non Financial Fees', 'Gastos operativos', 'Pérdida esperada', 'RWA', 'RORWA objetivo'], 'Valores':fix_params[fix_params.contrato == valor8].values[0]})
    
    fin_fees = ((valor7-valor6-valor5)*12*(tot['Saldo medio'].values[0]*(valor3-valor4))**-1)
    TII = valor2 + fin_fees
    TIE = -((ind.LS.values[0] + fix_params[fix_params.contrato == valor8].ITR.values[0]))
    non_fin_fees = -((fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0]))
    NII = TII + TIE
    GM = NII + non_fin_fees
    op_exp = -((fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0]))
    NOI = GM + op_exp
    EL = (fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0]*-1)
    PBT = NOI + EL
    PAT = 0.715*PBT
    RWA = fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1
    RORWA = (PAT*valor1 / RWA)


    p_and_l = np.round(pd.DataFrame({'Inputs':['TIN', 'Financial fees', 'Total Interest Income', 'Total Interest Expenses', 'Net Interest Income', 'Non Fiancial Fees', 'Gross Margin', 'Operating Expenses ', 'Net Operating Income', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'], 'Valores':[valor2, fin_fees, TII, TIE, NII, non_fin_fees, GM, op_exp, NOI, EL, PBT, PAT, RWA, RORWA]}), 2)
    p_and_l['Valores'] = [f"{np.round(valor*100, 2)}%" if i != p_and_l.shape[0]-2 else valor for i, valor in enumerate(p_and_l['Valores'])]

    by_monto = [getIndicators(i, valor2, valor3, valor4, valor5, valor6, valor7, fix_params[fix_params.contrato == valor8].ITR.values[0], fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0], fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0], fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0], fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1).RORWA.values[0] for i in np.arange(100, 50000, 10000)]
    by_TIN = [getIndicators(valor1, i, valor3, valor4, valor5, valor6, valor7, fix_params[fix_params.contrato == valor8].ITR.values[0], fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0], fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0], fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0], fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1).RORWA.values[0] for i in np.arange(-1.1, 1.1, 0.01)]
    by_pagos = [getIndicators(valor1, valor2, i, valor4, valor5, valor6, valor7, fix_params[fix_params.contrato == valor8].ITR.values[0], fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0], fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0], fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0], fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1).RORWA.values[0] for i in range(valor4+1, 70)]
    by_fapertura = [getIndicators(valor1, valor2, valor3, valor4, valor5, valor6, i, fix_params[fix_params.contrato == valor8].ITR.values[0], fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0], fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0], fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0], fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1).RORWA.values[0] for i in np.arange(0, 20000, 2000)]
    by_fterceros = [getIndicators(valor1, valor2, valor3, valor4, i, valor6, valor7, fix_params[fix_params.contrato == valor8].ITR.values[0], fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0], fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0], fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0], fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1).RORWA.values[0] for i in np.arange(0, 20000, 2000)]
    by_frappels = [getIndicators(valor1, valor2, valor3, valor4, valor5, i, valor7, fix_params[fix_params.contrato == valor8].ITR.values[0], fix_params[fix_params.contrato == valor8]['Comisiones no financieras'].values[0], fix_params[fix_params.contrato == valor8]['Gastos de explotacion'].values[0], fix_params[fix_params.contrato == valor8]['Perdida esperada'].values[0], fix_params[fix_params.contrato == valor8].RWA.values[0]*valor1).RORWA.values[0] for i in np.arange(0, 20000, 2000)]

    
    TIN_df = pd.DataFrame({'TIN':np.arange(-1.1, 1.1, 0.01), 'RORWA':by_TIN})
    pagos_df = pd.DataFrame({'Pagos':np.arange(valor4+1, 70), 'RORWA':by_pagos})
    fapertura_df = pd.DataFrame({'Comision Apertura':np.arange(0, 20000, 2000), 'RORWA':by_fapertura})
    fterceros_df = pd.DataFrame({'Comision terceros':np.arange(0, 20000, 2000), 'RORWA':by_fterceros})
    frappels_df = pd.DataFrame({'Comision rappels':np.arange(0, 20000, 2000), 'RORWA':by_frappels})
    monto_df = pd.DataFrame({'Monto':np.arange(100, 50000, 10000), 'RORWA':by_monto})

    
    fig1 = px.line(TIN_df, x='TIN', y='RORWA', height=300, width=500)

    fig2 = px.line(pagos_df, x='Pagos', y='RORWA', height=300, width=500)
    fig3 = px.line(fapertura_df, x='Comision Apertura', y='RORWA', height=300, width=500)
    fig4 = px.line(fterceros_df, x='Comision terceros', y='RORWA', height=300, width=500)
    fig5 = px.line(frappels_df, x='Comision rappels', y='RORWA', height=300, width=500)
    fig6 = px.line(monto_df, x='Monto', y='RORWA', height=300, width=500)
    

    params = fix_params.copy()

    return tot.to_dict('records'), tab.to_dict('records'), ind.to_dict('records'), fig1, fig2, fig3, p_and_l.to_dict('records'), fig4, fig5, fig6, params.to_dict('records')

if __name__ == "__main__":
    app.run_server(debug=False, port=8050)
