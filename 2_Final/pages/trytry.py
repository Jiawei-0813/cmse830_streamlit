categorical = st.multiselect(
        'Choose the categorical columns: ',
        ['season', 'holiday', 'workingday', 'weather'],
        ['season', 'holiday', 'workingday'])

    numerical = st.multiselect(
        'Choose the numerical columns: ',
        ['temp', 'atemp', 'humidity', 'windspeed'],
        ['temp', 'humidity', 'windspeed']
    )

    target = st.selectbox(
    'Choose the target: ',
    ('casual', 'registered', 'count'))

    '''
    ### Train-Test Split

    '''
    X = cf[categorical + numerical]
    y = cf[target]

    test_size = st.slider('', 0.01, 1.0, 0.15, 0.01)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    f'''
    X_train: {X_train.shape}  
    X_test: {X_test.shape}  
    y_train: {y_train.shape}  
    y_test: {y_test.shape}  
    '''
    
    

    f'''
    ## Data Preprocessing

    1.  {categorical} are categorical data so we can use one-hot encoding.
    1.  {numerical} are numerical data so we standardize them using standard scaler.

    '''

    X_train_transformed, ohe, sc = transform(X_train, fit=True)
    X_test_transformed , _, _ = transform(X_test, ohe=ohe, sc=sc)

    st.session_state['stuff'] = categorical, numerical, target, ohe, sc

    f'''
    After the transformation we get something like this:
    '''
    X_train_transformed

    st.session_state['data'] = X_train_transformed, X_test_transformed, y_train, y_test

if chapter == 'Linear Regression':

    '''
    ## Linear Regression
    '''

    X_train_transformed, X_test_transformed, y_train, y_test = st.session_state['data']

    model = LinearRegression()

    model.fit(X_train_transformed, y_train)

    st.session_state["Linear Regression"] = model

    prediction = model.predict(X_train_transformed)
    train_err = mean_squared_error(y_train, prediction, squared=False)
    
    f'''
    ### Prediction on Train set:
    MSE: {train_err:.5}  
    R2 Score: {r2_score(y_train, prediction):.5}
    '''

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(y_train, prediction, 'o')
    ax.set_title("Prediction on X_train", color='white')
    ax.set_xlabel('y_train', color='white')
    ax.set_ylabel('prediction', color='white')

    # Settings
    fig.set_facecolor("#0E1117")
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    prediction = model.predict(X_test_transformed)
    test_err = mean_squared_error(y_test, prediction, squared=False)

    f'''
    ### Prediction on Test set:
    MSE: {test_err:.5}  
    R2 Score: {r2_score(y_test, prediction):.5}
    '''

    ax = fig.add_subplot(1,2,2)
    ax.plot(y_test, prediction, 'o')
    ax.set_title("Prediction on X_test", color='white')
    ax.set_xlabel('y_test', color='white')

    # Settings
    fig.set_facecolor("#0E1117")
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    st.pyplot(fig)





if chapter == 'Neural Network':

    '''
    ## Neural Network
    '''

    X_train_transformed, X_test_transformed, y_train, y_test = st.session_state['data']

    solver = st.selectbox(
        'Select Solver: ',
        ('adam', 'sgd', 'lbfgs')
        )

    activation = st.selectbox(
        'Select activation function: ',
        ('relu', 'tanh', 'logistic', 'identity')
    )

    n_layers = st.slider('Number of hidden layers: ',
        1, 5, 2, 1)

    hidden_layer_sizes = [1]*n_layers

    for layer in range(n_layers):
        hidden_layer_sizes[layer] = st.slider(f"Number of neurons in layer {layer+1}",
                                                1, 20, max(7*(2-layer), 5), 1)

    model = MLPRegressor(alpha=1e-4, 
                        activation=activation,
                        solver=solver,
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=10000)

    model.fit(X_train_transformed, y_train)

    st.session_state["Neural Network"] = model

    prediction = model.predict(X_train_transformed)
    train_err = mean_squared_error(y_train, prediction, squared=False)
    
    f'''
    ### Prediction on Train set:
    MSE: {train_err:.5}  
    R2 Score: {r2_score(y_train, prediction):.5}
    '''

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(y_train, prediction, 'o')
    ax.set_title("Prediction on X_train", color='white')
    ax.set_xlabel('y_train', color='white')
    ax.set_ylabel('prediction', color='white')

    # Settings
    fig.set_facecolor("#0E1117")
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    prediction = model.predict(X_test_transformed)
    test_err = mean_squared_error(y_test, prediction, squared=False)

    f'''
    ### Prediction on Test set:
    MSE: {test_err:.5}  
    R2 Score: {r2_score(y_test, prediction):.5}
    '''

    ax = fig.add_subplot(1,2,2)
    ax.plot(y_test, prediction, 'o')
    ax.set_title("Prediction on X_test", color='white')
    ax.set_xlabel('y_test', color='white')

    # Settings
    fig.set_facecolor("#0E1117")
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    st.pyplot(fig)


if chapter == 'Kernel Ridge Regression':

    '''
    ## Kernel Ridge Regression
    '''

    X_train_transformed, X_test_transformed, y_train, y_test = st.session_state['data']

    kernel = st.selectbox(
        'Select Kernel: ',
            ['rbf','polynomial','cosine','sigmoid','linear']
        )

    alpha_num = st.slider('Alpha Number: ',
        1, 999, 1, 1)


    alpha_exp = st.slider('Alpha Exponent: ',
        -10, 1, -7, 1)

    gamma_num = st.slider('Gamma Number: ',
        1, 999, 543, 1)


    gamma_exp = st.slider('Gamma Exponent: ',
        -10, 1, -6, 1)


    model = KernelRidge(kernel=kernel,
                    alpha=alpha_num*(10**alpha_exp),
                    gamma=gamma_num*(10**gamma_exp)
    )

    model.fit(X_train_transformed, y_train)

    st.session_state['Kernel Ridge'] = model

    prediction = model.predict(X_train_transformed)
    train_err = mean_squared_error(y_train, prediction, squared=False)
    
    f'''
    ### Prediction on Train set:
    MSE: {train_err:.5}  
    R2 Score: {r2_score(y_train, prediction):.5}
    '''

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(y_train, prediction, 'o')
    ax.set_title("Prediction on X_train", color='white')
    ax.set_xlabel('y_train', color='white')
    ax.set_ylabel('prediction', color='white')

    # Settings
    fig.set_facecolor("#0E1117")
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    prediction = model.predict(X_test_transformed)
    test_err = mean_squared_error(y_test, prediction, squared=False)

    f'''
    ### Prediction on Test set:
    MSE: {test_err:.5}  
    R2 Score: {r2_score(y_test, prediction):.5}
    '''

    ax = fig.add_subplot(1,2,2)
    ax.plot(y_test, prediction, 'o')
    ax.set_title("Prediction on X_test", color='white')
    ax.set_xlabel('y_test', color='white')

    # Settings
    fig.set_facecolor("#0E1117")
    ax.tick_params(axis='both', colors='#ffffff')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#0E1117')

    st.pyplot(fig)


if chapter=="User Input":

    f'''
    ## Make Predictions Live

    The best model results arise from using the Kernal Ridge Regressor with the given default values.  
    However, feel free to choose the model you'd like to make the prediction with:
    '''

    model_name = st.selectbox(
        'Select Model: ',
        ( 'Kernel Ridge', 'Neural Network', 'Linear Regression')
        )

    model = st.session_state[model_name]
    categorical, numerical, target, ohe, sc = st.session_state['stuff']

    weats = ['Cloudy', 'Light Rain', 'Heavy Rain']
    seas = ['Spring', 'Summer', 'Fall', 'Winter']
    boo = ['No', 'Yes']

    df = {}

    for cat in categorical:
        if cat == 'season':
            thing = seas.index(st.selectbox("Season:", seas))+1

        if cat == 'holiday':
            thing = boo.index(st.selectbox("Holiday:", boo))

        if cat == 'workingday':
            thing = boo.index(st.selectbox("Working Day:", boo))

        if cat=='weather':
            thing = weats.index(st.selectbox("Weather:", weats))+1

        df[cat] = [thing]

    for num in numerical:
        thing = st.slider(num.capitalize(), cf[num].min(), cf[num].max(), 0.1)
        df[num] = [thing]

    X = pd.DataFrame(df)
    
    '### Data:'
    st.write(X)

    X_trans, _, _ = transform(X, ohe, sc)

    '### Transformed Data:'
    st.write(X_trans)
    try:
        f'''
        ### Prediction:
        
        Today, you will have {model.predict(X_trans)[0]:.0f} {target} users per hour.

        '''
    except:
        f'''
        #### --- Train the model, then try again ---
        '''