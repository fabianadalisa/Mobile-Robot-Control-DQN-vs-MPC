function [InitialObservation, LoggedSignals] = my_resetfunction()
% RESET FUNCTION adattata al modello con mappa, tavoli ,
% inizializza ogni volta lo stato in ogni episodio successivo

%% ---------------- Parametri iniziali ----------------
W = 20; H = 20;        % dimensione mappa (colonne x righe)
goal_tol = 1;        % tolleranza raggiungimento goal
Ts = 3;              % passo di simulazione [s]
%num_step=80;  % numero massimo di step per episodio

%% ---------------- Creazione mappa ----------------
map = zeros(H,W);
map(1,:) = 1; map(end,:) = 1; map(:,1) = 1; map(:,end) = 1; % muri perimetrali

% Tavoli statici [row col]
 tables = [5 5; 8 10; 12 8; 15 15];

 for k = 1:size(tables,1)
        r = tables(k,1);
        c = tables(k,2);
        map(r:r+1, c:c+1) = 1;   % blocchi 2x2
 end

%% ---------------- Posizione iniziale robot ----------------
sx = 4; sy = 8; theta = pi/4; % partenza fissa in [2,2] orientato a pi/2

State = [sx; sy; theta];  % stato iniziale 3x1

% Normalizzazione stato iniziale
x_norm     = sx/ W;
y_norm     = sy / H;
theta_norm = theta / pi;


%% ---------------- Goal fisso ----------------
goal = [15;18];  % obiettivo definito in cella [15,15]

%% ---------------- LoggedSignals ----------------%servono per inizzializzare e per poter definire reward nella step
LoggedSignals = struct();
LoggedSignals.State = State;       % stato iniziale
LoggedSignals.curr_time = 1;       % tempo corrente
LoggedSignals.num_steps = 120; 
LoggedSignals.map = map;           % mappa
LoggedSignals.W = W;               
LoggedSignals.H = H;
LoggedSignals.tables = tables;     
LoggedSignals.Ts = Ts;             
LoggedSignals.goal = goal;         
LoggedSignals.goal_tol = goal_tol;

%% ---------------- Osservazione iniziale ----------------
InitialObservation = [x_norm; y_norm; theta_norm];


end