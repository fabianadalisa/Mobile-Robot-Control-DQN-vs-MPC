function [NextObs,Reward,IsDone,LoggedSignals] = my_stepfunction(Action,LoggedSignals)

% funzione che viene richiamata ad ogni passo di simulazione, riceve come
% input l'action,cioè l 'azione scelta dall'agente e la struttura logged
% signal costituita dallo stato e da altri dati relativi all'ambiente.
% Restituisce come output nextobs cioè il nuovo stato del
% robot, la reward, la is done condition, se finisce resttuisce 1 o
% altrimenti 0, e in fine la nuova strutttura logged signal


% STEP FUNCTION con goal fisso [15,15]

%% ---------------- Estrazione stato e parametri ----------------
x_k = LoggedSignals.State;          % stato corrente [x; y; theta] 
kk = LoggedSignals.curr_time;       % tempo corrente (step index)
Ts = LoggedSignals.Ts;              % passo di simulazione, tempo di campionamento
num_step = LoggedSignals.num_steps;   % numero passi per episodio
map = LoggedSignals.map;            % mappa con muri/tavoli
W = LoggedSignals.W; 
H = LoggedSignals.H;

% definiamo una struttura anche per il Goal
goal = LoggedSignals.goal;        % usa il goal  dalla reset
goal_tol = LoggedSignals.goal_tol; % tolleranza

% LoggedSignals è una struttura (struct) che contiene tutte le informazioni che vuoi conservare tra uno step e l’altro dell’ambiente di Reinforcement Learning.
% Dentro questa struttura c’è un campo chiamato goal, che di solito contiene:
% la posizione finale che il robot deve raggiungere (es. [x_goal, y_goal])
% oppure lo stato obiettivo completo.


%% ---------------- Traduzione azione ----------------

% Definiamo le azioni in input del robot 
if isscalar(Action)
  switch Action

    % --- 1) Movimento avanti ---
    case 1,  v_cmd = 0.15; w_cmd = 0;          % avanti molto lento
    case 2,  v_cmd = 0.30; w_cmd = 0;          % avanti lento
    case 3,  v_cmd = 0.45; w_cmd = 0;          % avanti medio

    % --- 2) Movimento indietro ---
    case 4,  v_cmd = -0.15; w_cmd = 0;         % indietro lento
    case 5,  v_cmd = -0.30; w_cmd = 0;         % indietro medio

    % --- 3) Rotazioni pure sinistra ---
    case 6,  v_cmd = 0; w_cmd =  pi/16;        % sinistra piccola
    case 7,  v_cmd = 0; w_cmd =  pi/8;         % sinistra media
    case 8,  v_cmd = 0; w_cmd =  pi/6;         % sinistra larga

    % --- 4) Rotazioni pure destra ---
    case 9,  v_cmd = 0; w_cmd = -pi/16;        % destra piccola
    case 10, v_cmd = 0; w_cmd = -pi/8;         % destra media
    case 11, v_cmd = 0; w_cmd = -pi/6;         % destra larga

    % --- 5) Curve combinate ---
    case 12, v_cmd = 0.30; w_cmd =  pi/16;     % curva lieve a sinistra
    case 13, v_cmd = 0.30; w_cmd = -pi/16;     % curva lieve a destra

    % --- Stop ---
    case 14, v_cmd = 0; w_cmd = 0;

    otherwise
        v_cmd = 0; w_cmd = 0;
 end
  
end 

% Comando finale sempre come vettore colonna
action_vec = [v_cmd; w_cmd];

%% ---------------- Integrazione dinamica robot ----------------

tspan = [kk*Ts , (kk+1)*Ts]; %è l'intervallo di tempo continuo da adesso al prossimo istante discreto
                             % si usa ovviamente perchè ode45 opera su
                             % intrvalli di tempo continui
% simulo l'evoluzione dinamica del sistema
    [~, x_temp] = ode45(@(t,x)my_pitch(t,x,action_vec), tspan, x_k);
    x_next = x_temp(end,:)';% così facendo stiamo prendendo lo stato finale dopo
    %aver integrato per un tempo ts e stiamo costruendo una matrice dei
    %nostri 3 stati negli istanti finali di ts nei vari intervalli continui
    %successivi

% aggiungi piccolo rumore, per rendere più robusto e realistico
% l'apprendimento
x_next(1:2) = x_next(1:2) + 0.01*randn(2,1);
x_next(3) = wrapToPi(x_next(3)); %serve per aggiungere rumore a theta non si può fare con
%rand perchè poichè l'angolo è in radianti potrebbero uscure valori di
%angoli non validi


%% ---------------- Normalizzazione stato ----------------
% Stato normalizzato rispetto alla mappa 20x20 e all'intervallo angolare
x_norm     = x_next(1) / W;      % x in [0,1]
y_norm     = x_next(2) / H;      % y in [0,1]
theta_norm = x_next(3) / pi;     % theta in [-1,1]

state_norm = [x_norm; y_norm; theta_norm];

% Goal normalizzato (stesso sistema di riferimento)
goal_norm  = [goal(1)/W; goal(2)/H; 0];

% Errore normalizzato
error_norm = state_norm - goal_norm;

%% --------- Distanza e progresso verso il goal (coordinate reali) ---------
dist_prev = norm(x_k(1:2)    - goal);    % distanza al passo k
dist_now  = norm(x_next(1:2) - goal);    % distanza al passo k+1

progress  = dist_prev - dist_now;        % >0 se ti avvicini


%% ---------------- Collisioni con ostacoli statici ----------------
% Converti lo stato continuo in indici di cella, il comando round converte
% lo stato che con ogni probabilità sarà un numero decimale nell'intero più
% vicino
col_idx = round(x_next(1));
row_idx = round(x_next(2));

% Clamp ai limiti della mappa
col_idx = min(max(col_idx, 1), W);
row_idx = min(max(row_idx, 1), H); % Garantisce che gli indici:
% non vadano sotto 1
% non vadano sopra W (colonne) o H (righe) serve perchè il robot ha
% coordinate continue che possono essere leggermente fuori dai confini
% definiti a causa di rumore o arrotondamenti

% Controllo cella mappa
if map(row_idx, col_idx) == 1
    Reward = -80; 
    IsDone = 1; %episodio termina
    NextObs = state_norm; %mostro all'agente la posizione in cui ha colpito ostacolo
    LoggedSignals.State = x_next; %aggiorno stato interno dell'ambiente alla poszione
    %della collisione perchè la step function deve sempre aggiornare lo
    %stato prima di uscire
    LoggedSignals.curr_time = kk+1;
    
    return; %blocca la stepfunction episodio finito
end

%% ---------------- Goal check ----------------
dist_to_goal = norm([x_next(1),x_next(2)]-goal');%norm calcola la distanza euclidea tra il goal e la posizione del robot
if dist_to_goal< goal_tol
    Reward = +200;
    IsDone = 1;
    NextObs =state_norm;
    LoggedSignals.State = x_next;
    LoggedSignals.curr_time = kk+1;
   
    return;
end


%% ---------------- Ricompensa normale ----------------
%dist_to_goal = norm(x_next(1:2) - goal); %stessa riga di sopra scritta in modo più compatto
%Reward = -dist_to_goal - 0.1*norm(action_vec);  % diamo bonus negatiovo direttamnete proporzionale alla distanza dal goal

% Bonus per avvicinamento al goal
%prev_dist = norm(x_k(1:2) - goal');   % distanza precedente
%progress = prev_dist - dist_to_goal;  % se la distanza precedente dal goal è maggiore di distanza attuale da goal ci siamo avvicinati diamo premio
%Reward = Reward + 0.5 * progress;
%% ---------------- Ricompensa normale (Sostituzione con Q/R) ----------------
% Definizioni necessarie (da mettere all'inizio della step function)
Q = diag([0.5, 0.5, 0.1]); % Penalità stato: [x, y, theta]
R = diag([0.01, 0.02]); % Penalità azione: [v, w]

time_penalty = -0.5;
% 2. Calcola la Penalità Quadrata (Cost function)
Reward_penalita = error_norm' * Q * error_norm + action_vec' * R * action_vec;
% 3. La Reward è l'opposto della penalità
Reward =+time_penalty+ 10*progress-Reward_penalita;

%% ---------------- Overflow threshold ----------------
State_threshold = 20; %è fondamentale anche se c'è già il clamp perchè quest'ultimo non modifica lo stato in se ma gli indici
%esempio col_idx = 1 → OK
%        row_idx = H → OK
%        ma x_next = [-3.2; 25.7; 0.4] è ancora fuori mappa
% quindi il threshold evita stati impossibili chiudendo episodio
if max(abs(x_next))>State_threshold %se il valore assoluto di una delle 2 variabili è piè grande del limite quindi la posizione 
    %è fuori dallo spazio delimitato  allora succede quello che c'è scritto
    %qua sotto
    Reward = -80; 
    IsDone = 1;
    NextObs = state_norm;
    LoggedSignals.State = x_next;
    LoggedSignals.curr_time = kk+1;
    return;
end
%% ---------------- Check numero massimo di step ----------------
if kk >= num_step
    % Penalità finale proporzionale alla distanza dal goal
    dist_to_goal = norm(x_next(1:2) - goal);
    Reward = -5*dist_to_goal;
    IsDone = 1;
    NextObs = state_norm;
    LoggedSignals.State = x_next;
    LoggedSignals.curr_time = kk+1;

    return;
end

%% ---------------- Aggiornamento finale ----------------
IsDone = 0;
NextObs = state_norm;
LoggedSignals.State = x_next;
LoggedSignals.curr_time = kk+1;

%serve per far continuare la simulazione fino a quando non si verifica uno
%dei casi precedenti