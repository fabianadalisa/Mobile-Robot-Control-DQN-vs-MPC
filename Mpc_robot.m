clear;
clc;
close all;
%% ---------------- Creazione mappa ----------------
W = 20; H = 20;        % dimensione mappa (colonne x righe)
map = zeros(H,W);
map(1,:) = 1; map(end,:) = 1; map(:,1) = 1; map(:,end) = 1; % muri perimetrali
% Tavoli statici rappresentano gli ostaoli all'interno del nostro warehouse
 tables = [5 5; 8 10; 12 8; 15 15];
 for k_map = 1:size(tables,1)
        r = tables(k_map,1);
        c = tables(k_map,2);
        map(r:r+1, c:c+1) = 1;   % blocchi 2x2
 end

%% Parametri del controllo
T = 800; % Tempo totale di simulazione
T_c = 20; % Orizzonte di controllo, definisce la lunghezza dell'intervallo di tempo della previsione con cui il robot pianifica la sua traiettoria  
n = 3; % Numero di stati [x, y, theta]
m = 2; % Numero di ingressi di controllo [v, omega]

% Parametri del modello Cinematico (utilizzati per linearizzare)
v_r = 0.35; % Velocità lineare di riferimento (usata per B_k)
T_s = 0.05; % Periodo di campionamento

% Stato di riferimento (Obiettivo da raggiungere)
x_ref = [15; 18; 0]; % Obiettivo: X=15, Y=18, Theta=0

% Q e R Matrici di costo
Q = diag([10, 10, 0.5]); % Matrice di costo sugli stati
% Q penalizza la deviazione dello stato previsto dal robot, dallo stato di
% riferimento X ref, in questo caso andiamo a penalizzare maggiormente
% l'errore di posizione rispetto all'errore di orientamento
R = 0.1 * eye(m); % Matrice di costo sugli ingressi
% R invece penalizza l'energia  e lo sforzo di controllo, evitando che il
% robot possa chiedere input troppo grandi irregolari o oscillanti, nel
% nostro caso i pesi sia su v che w sono uniformi e molto meno penalizzanti
% rispetto a quelli in Q
Q_f = Q; % Matrice di costo terminale

x(:, 1) = [4; 8; pi/4]; % Stato iniziale
% scriviamo i limiti agli ingressi dati nel paper 
u_min = [-0.47; -3.3]; % Limiti inferiori [v_min; omega_min]
u_max = [0.47; 3.3]; % Limiti superiori [v_max; omega_max]

%% Simulazione e Ottimizzazione MPC
figure;
hold on;
% Fissa i limiti degli assi per la visualizzazione della mappa
axis([1 W 1 H]); 
title('Robot in Magazzino - LTV-MPC');
xlabel('X');
ylabel('Y');
grid on;

% Placeholder per simboli corretti nella legenda
h_fixed = plot(NaN, NaN, 's', 'Color', [0, 0, 0], 'MarkerFaceColor', [0, 0, 0]);
h_robot = plot(NaN, NaN, 'o', 'Color', [0.1, 0.1, 0.8], 'MarkerFaceColor', [0.1, 0.1, 0.8]);
h_goal = plot(NaN, NaN, 'o', 'Color', [1, 0, 0], 'MarkerFaceColor', [1, 0, 0]);
h_trajectory = plot(NaN, NaN, '-', 'Color', [0, 0, 1]);
legend([h_fixed, h_robot, h_goal, h_trajectory], ...
       {'Ostacoli Fissi', 'Robot', 'Goal', 'Traiettoria'}, 'Location', 'northeast');

for k = 1:T-1    % usiamo il ciclo for per iniziare la simulazione
   % Visualizzazione del magazzino
   cla; % clear axes, serve a creare l'effetto di animazione

   % ---------------- LTV-MPC: LINEARIZZAZIONE ALLO STATO CORRENTE ----------------
   
   % Riferimento di linearizzazione: Orientazione attuale del robot
   theta_k = x(3, k); % è fondamentale perchè invece di avere un orientazione fissa in questo modo avremo 
   % l'orientazione allo stato attuale del robot, ed è fondamentale per
   % linearizzare e cambiare in maniera continua ad ogni passo A e B, in
   % quanto sia il seno che il coseno dipendono da k
   
   % Le matrici A e B cambiano ad ogni passo (A_k e B_k)
   A_k = [1, 0, -v_r * sin(theta_k) * T_s;
         0, 1,  v_r * cos(theta_k) * T_s;
         0, 0,  1]; % Matrice di stato dipendente da theta_k
         
   B_k = [cos(theta_k) * T_s, 0;
         sin(theta_k) * T_s, 0;
         0,                  T_s]; % Matrice di controllo dipendente da theta_k
   % in questo modo andiamo ad aggiornare il controller dopo ogni istante
   % trasformando un problema non lineare difficile in una serie di
   % problemi lineari semplici, consentendo un controllo efficace
         
   % ---------------- FINE LTV-MPC LINEARIZZAZIONE ----------------
   % NB = siamo sempre all'interno del ciclo for principale
   % Visualizzazione degli ostacoli fissi dalla matrice 'map'
   imagesc(map);  % visualizziamo la mappa come un immagine
   colormap([1 1 1; 0 0 0]); % Colori: 0=Bianco (libero), 1=Nero (ostacoli e mura)
   axis([1 W 1 H]);  % fissa i limiti degli assi andando da 1 a 20
   set(gca, 'YDir', 'normal'); % Allinea l'asse Y con la convenzione cartesiana
 
   %%---------------- Controllo MPC -------------------------------
   % Si passa A_k e B_k alla funzione di costo
   cost_function = @(u) my_MPC_costfunction(u, A_k, B_k, x(:, k), x_ref, T_c, Q, Q_f, R, map, W, H);
   % la cost function rappresenta la funzione obiettivo che deve essere
   % minimizzata, fondamentale come notiamo da @ è che solo il vettore u è
   % la variabile di ottimizzazione
   u_guess = zeros(m * T_c, 1); % Ipotesi iniziale dell'ingresso al primo istante
   options = optimoptions('fmincon', 'Display', 'off'); % configura il solver fmincon
   lb = repmat(u_min, T_c, 1); 
   ub = repmat(u_max, T_c, 1);
   % aggiorniamo sempre ad ogni passo anche i limiti inferiori e superiori
   
   % OTTIMIZZAZIONE
   u_opt = fmincon(cost_function, u_guess, [], [], [], [], lb, ub, [], options);
   % questa è la riga che esegue l'ottimizzazione , riceve la funzione di
   % costo,l'ipotesi iniziale  e i limiti e calcola l'intera sequenza di
   % ingressi di controllo ottimali Uopt per l'orizzonte Tc=20

   % Applica il primo ingresso ottimale
   u_k = u_opt(1:m); % in questo modo l'MPC utilizza solo il primo vettore di controllo della sequenza ottimale,
   % gli input successivi vengono scartati perche ricalcolati
   % all'istante k+1
   
   % PROPAGAZIONE DELLO STATO REALE (USA A_k e B_k)
   x(:, k+1) = A_k * x(:, k) + B_k * u_k;  % calcolo del nuovo stato del robot
   
   % Disegna il robot (con marker semplice per coerenza con imagesc)
   plot(x(1, k), x(2, k), 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.1, 0.1, 0.8], 'MarkerEdgeColor', 'k');
   
   % Disegna il goal
   plot(x_ref(1), x_ref(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
   
   % Pausa per visualizzazione
   pause(0.05);
end  % chiudiamo il ciclo for k=1:T-1

% Disegna la traiettoria completa del robot
plot(x(1, :), x(2, :), 'b-', 'LineWidth', 3);


%% Funzione di costo (LTV + Collisione a Penalità)
function J = my_MPC_costfunction(u, A, B, x0, x_ref, T_c, Q, Q_f, R, map, W, H)
 % definiamo la funzione di costo che riceve la sequenza di u e tutti i
 % parametri fissi, J è il risultato scalare finale, ed fmincon lavora per
 % rendere J quanto più piccolo possibile

m = size(B, 2);   % calcola il numero di ingressi di controllo, in questo caso 2
   x = x0; 
   J = 0;    % inizializza lo stato per la previsione con lo stato attuale misurato     
   
   % AUMENTA QUESTA PENALITÀ se il robot continua a schiantarsi!
   J_collision = 1e8; % Costo di penalità molto alto per penalizzare qualsiasi collisione prevista

   % Inizia adesso il ciclo di previsione sull'orizzonte di controllo
   for t = 1:T_c
       u_t = u((t-1)*m+1:t*m);
       
       % Previsione dello Stato (usa le matrici A e B del passo k)
       x = A * x + B * u_t; 
       
       % ---------------- LOGICA DI COLLISIONE ----------------
       
       % 1. Conversione delle coordinate continue [x, y] in indici di cella [riga, colonna]
       r = round(x(2)); 
       c = round(x(1)); 
       
       % 2. Saturazione (necessaria per rimanere nei limiti della matrice [1, H] e [1, W])
       r = max(1, min(H, r));
       c = max(1, min(W, c));
       
       % 3. Check e Penalità
       if map(r, c) == 1
           % Se la cella prevista è un ostacolo, aggiunge il costo enorme
           J = J + J_collision;  % avremo quindi un costo enorme che porterà il robot ad evitarlo
       end
       
       % ---------------- FINE LOGICA ----------------
       
       % Costi standard (Quadratic Cost) % aggiunge i costi standard ad
       % ogni passo
       J = J + (x - x_ref)' * Q * (x - x_ref) + u_t' * R * u_t;
   end 
   
   % Costo Terminale, si calcola quando il ciclo di previsione è finito, si
   % aggiunge il costo utilizzando la matrice di peso temrinale Qf, questo
   % incoraggia la traiettoria ad avvicinarsi il più possibile
   % all'obiettivo
   J = J + (x - x_ref)' * Q_f * (x - x_ref);
end