clear
clc
close all

%% Ambiente RL: osservazioni e azioni discrete scalari
n=3;
obsInfo = rlNumericSpec([n 1]); %qui stiamo solo dicendo al toolbox che l'agente riceverà un vettore continuo di n=e variabili
obsInfo.Name = 'Robot States';
obsInfo.Description = 'x,y,theta';

%v= 0:.4:6;
%w= 0:pi/10:2*pi;
%m_1= length(v);
%m_2= length(w);

%for jj = 1:m_1 
    %for kk = 1:m_2 
     %   my_cell{jj,kk} = [v(jj) w(kk)]; 
    %end 
%end   %poichè noi abbiamo definito delle action nella step richiamerememo
%quelle


actInfo = rlFiniteSetSpec(1:14);  % azioni discrete: avanti, ruota sx, ruota dx, stop
actInfo.Name = 'Discrete Action';
 
%env = rlFunctionEnv(obsInfo, actInfo, 'my_stepfunction', 'my_resetfunction')';%creo l'ambiente
%obsInfo = getObservationInfo(env);
%actInfo = getActionInfo(env);
% Reshapes the cell so that the first dimension is 1
%my_cell = reshape(my_cell,1,m_1*m_2);

%% Rete neurale sostituisce la Q-table perché lo stato è continuo,approssima Q(s,a) per ogni azione.
%% Ogni output della rete corrisponde a un Q per una delle tue 9 azioni.Senza rete non puoi fare Q-learning su stati continui.
net = [
    featureInputLayer(obsInfo.Dimension(1), Name="state")
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(length(actInfo.Elements))
];
net = dlnetwork(net);
summary(net) %controlla se gli imput sono corretti

% Grafico rete neurale
plot(net); 


%% Critic

%critic = rlQValueFunction(criticNetwork, obsInfo, actInfo);

criticOptions = rlOptimizerOptions( ...
    LearnRate=5e-4, ...
    GradientThreshold=1);
%% Ambiente RL
env = rlFunctionEnv(obsInfo, actInfo, 'my_stepfunction','my_resetfunction'); %l'ho creato prima 

 
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
critic = rlVectorQValueFunction(net,obsInfo,actInfo);

%% Agente DQN
agentOpts = rlDQNAgentOptions( ...
    'UseDoubleDQN', true, ...
    'TargetUpdateMethod', "periodic", ...
    'TargetUpdateFrequency', 50, ...
    'ExperienceBufferLength', 20000, ...
    'DiscountFactor', 0.99, ...
    'MiniBatchSize', 64, ...
    'CriticOptimizerOptions', criticOptions);

agentOpts.EpsilonGreedyExploration.Epsilon = 0.05;
% agentOpts.EpsilonGreedyExploration.Epsilon      = 1.0;   % inizio: esplora sempre
%agentOpts.EpsilonGreedyExploration.EpsilonMin   = 0.05;  % minimo: 5% di esplorazione
%agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.995; % decrescita graduale
%nell'episodi iniziali l'agente prova tutti man mano che si allena
%l'explorationscende e dopo un tot di episodi epsilon=0.05

agent = rlDQNAgent(critic, agentOpts);


%% Training
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 2000, ... 
    'MaxStepsPerEpisode', 120, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',500, ...
    UseParallel=false); 
trainOpts.SaveAgentCriteria = "AverageReward";
trainOpts.SaveAgentValue = -1000;

trainingStats = train(agent, env, trainOpts);

%% Simulazione post-training
simOpts = rlSimulationOptions;
simOpts.NumSimulations = 50;
simOpts.StopOnError = 'off';
experience = sim(env, agent, simOpts);

%x = squeeze(experience(10).Observation.RobotStates.Data(:,1,:));
%my_u = squeeze(experience(10).Action.DiscreteAction.Data(:,1,:));
%plot(x')

numGoals = 0;
totReward = zeros(50,1);

for ep = 1:50
    r = experience(ep).Reward.Data;
    totReward(ep) = sum(r);

    if experience(ep).Reward.Data(end) > 0   % ultimo reward positivo = goal
        numGoals = numGoals + 1;
    end
end

successRate = numGoals / 50
meanEpisodeReward = mean(totReward)
%nel mio caso la reward positiva comporta il raggiungimento del gol
%ovviamente non è una regola generale, ma valida a queste condizioni in
%cui l'unica reward positiva è quella del raggiungimento del goal


%% --- Parametri mappa (devono coincidere con la reset) ---
W = 20; H = 20;
map = zeros(H,W);

% muri perimetrali
map(1,:)   = 1;
map(end,:) = 1;
map(:,1)   = 1;
map(:,end) = 1;

% tavoli (stessi indici che usi nella reset)
tables = [5 5; 8 10; 12 8; 15 15];
for k = 1:size(tables,1)
    r = tables(k,1); c = tables(k,2);
    map(r:r+1, c:c+1) = 1;
end

goal = [15;18];      % stesso goal della reset

%% --- Figura mappa + tutte le traiettorie ---
figure; hold on; axis equal; grid on;
title('Traiettorie post-training su mappa 20x20');
xlabel('X'); ylabel('Y');
xlim([0 W]); ylim([0 H]);

% ---- Disegna ostacoli ----
for r = 1:H
    for c = 1:W
        if map(r,c) == 1
            rectangle('Position',[c-1, r-1, 1, 1], ...
                      'FaceColor',[0.7 0.7 0.7], ...
                      'EdgeColor','k');
        end
    end
end

% ---- Plotta tutte le traiettorie ----
nSim = numel(experience);

for ep = 1:nSim
    % stati [x;y;theta] per episodio ep
    x = squeeze(experience(ep).Observation.RobotStates.Data(:,1,:));
    x_real = x(1,:)*W;
    y_real = x(2,:)*H;
    % se per qualche motivo l’episodio è vuoto, skippa
    if isempty(x)
        continue;
    end

    % traiettoria
    plot(x_real, y_real, '-', 'LineWidth', 1);  % puoi aggiungere 'Color',[0 0 1 0.3] se hai Matlab recente

    % opzionale: start e fine (ma rischi di sporcare la figura se sono tanti episodi)
    % plot(x(1,1),   x(2,1),   'go', 'MarkerSize',4,'LineWidth',1);
    % plot(x(1,end), x(2,end), 'ro', 'MarkerSize',4,'LineWidth',1);
end

% ---- Plotta il goal una volta sola ----
scatter(goal(1), goal(2), 120, 'r', 'filled', 'DisplayName','Goal');

legend('Goal');  % le traiettorie non le metto in legenda per non fare casino


save('my_ws','agent','env','trainingStats')