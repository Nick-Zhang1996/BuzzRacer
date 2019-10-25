% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) OMG Plc 2009.
% All rights reserved.  This software is protected by copyright
% law and international treaties.  No part of this software / document
% may be reproduced or distributed in any form or by any means,
% whether transiently or incidentally to some other use of this software,
% without the written permission of the copyright owner.

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part of the Vicon DataStream SDK for MATLAB.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% FACTS:
fprintf('-----\n');
fprintf('VICON\n');
fprintf('-----\n');
fprintf('Make sure VICON software is on.\n');
fprintf('Ensure OBJECTS are selected in VICON software.\n');
fprintf('Ensure SOFTWARE is set to LIVE.\n');

%% VICON IP for tracker 192.168.10.1    
% The following assumes that there is a model running on xpc target
% with a handle "tg" and a constant block named "Constant"


% Settings:
TransmitMulticast = false;
 
% VICON Files:
addpath(genpath('VICON'));   %add folder that includes all necessary VICON files

% Load the SDK
fprintf( 'Loading SDK.' );
Client.LoadViconDataStreamSDK();
fprintf( 'Done loading SDK.\n' );

% Program options
HostName = 'localhost:801';

% Make a new client
MyClient = Client();

% Connect to a server:
fprintf( 'Connecting to server %s:\n', HostName );
while ~MyClient.IsConnected().Connected
  % Direct connection
  MyClient.Connect( HostName);
  fprintf( '.' );
end
fprintf( '\n' );

% Enable some different data types
MyClient.EnableSegmentData();
MyClient.EnableMarkerData();
MyClient.EnableUnlabeledMarkerData();
MyClient.EnableDeviceData();

% Checking frame rate
% MyClient.GetFrame();
% Output = MyClient.GetFrameRateName(1);
% ValueOutput = MyClient.GetFrameRateValue(Output.Name);
% Output.Result.Value
% Set the streaming mode
MyClient.SetStreamMode( StreamMode.ClientPull );

% Set the global up axis
MyClient.SetAxisMapping( Direction.Forward, ...
                         Direction.Left,    ...
                         Direction.Up );    % Z-up
                     
Output_GetAxisMapping = MyClient.GetAxisMapping();

% Discover the version number
Output_GetVersion = MyClient.GetVersion();


if TransmitMulticast
  MyClient.StartTransmittingMulticast( 'localhost', '224.0.0.0' );
end  


%---------------- Variable Initialization  --------------------------------
timestep     = 0;
time_vector  = zeros(timestep,1);
latency_store= zeros(timestep,1);

%--------------------- TCP/IP  ------------------------------------------
% SELECT synchronization with PC-104 for image grabbing or independent
% VICON operation
synch_with_pc104=0;  %0-> no synch with pc104, 1-> socket enabled

%--------------------- Initialize variable storage -----------------------
sframe_rotation_quat=[];
time_vector         =[];
time_vector_2       =[];
latency_store       =[];
sframe_store        =[];

%------------------------------------------------------------------------


% Define objects to be queried:
SubjectName_2   ='sframe'          ;SegmentName_2   = 'sframe'     ;
SubjectName ='porsche_barechassis';              SegmentName = 'porsche_barechassis';

% Establish UDP connection to target PC to transmit VICON data
udp_object = udp('192.168.7.12',3883,'ByteOrder','littleEndian'); %xPC "UPD Receive Binary" block expects bytes in different order than sent by default by "fwrite". 
%set(udp_object, 'OutputBufferSize', 7*8)                          ; %Specifies the maximum number of bytes that can be written to the server at once. 1 double = 8 bytes
fopen(udp_object);
%StopTime = getparam(tg,'StopTime');
StopTime = 100;
% Set up duration of experiment:
experiment_duration = StopTime+2;
fprintf('VICON_driver will run for %.3f s.\n',experiment_duration);
sframe_rotation_quat = zeros(4,experiment_duration*100);

timestep=0;
start_time_tg     = tic;
t = toc(start_time_tg);
while (t<experiment_duration) %(timestep<horizon)
    % Print to cmd window the time step

    while(mod( floor(t*1000), 100)~=0) % millisecond precision on query timing
        t = toc(start_time_tg);
    end
    timestep = timestep+1;
       % do nothing
    if mod(timestep,100) ==0
       fprintf('%d\n', timestep/100);
    end
    
%     % I don't know what comment to add here, but I would say: "wait for
%     % VICON to get a good measurement"
     while MyClient.GetFrame().Result.Value ~= Result.Success
%     
     end
    
    % Query both translation and rotation:
    Output_GetSegmentGlobalTranslation        = MyClient.GetSegmentGlobalTranslation(        SubjectName, SegmentName );
    Output_GetSegmentGlobalRotationQuaternion = MyClient.GetSegmentGlobalRotationQuaternion( SubjectName, SegmentName );
    
    % Extract translation:
    Global_Chaser_Translation = Output_GetSegmentGlobalTranslation.Translation; 

    % Code to avoid discontinuities in quaternion measurement          
    sframe_rotation_quat_original  =  Output_GetSegmentGlobalRotationQuaternion.Rotation;
    sframe_rotation_quat_symmetric = -sframe_rotation_quat_original                     ;
    if timestep>1
        if norm(sframe_rotation_quat_symmetric-sframe_rotation_quat(:,timestep-1))<norm(sframe_rotation_quat_original-sframe_rotation_quat(:,timestep-1))
            sframe_rotation_quat(:,timestep) = sframe_rotation_quat_symmetric;
        else
            sframe_rotation_quat(:,timestep)   = sframe_rotation_quat_original;
        end
    else
        sframe_rotation_quat(:,timestep) = sframe_rotation_quat_original;
    end    

    % Prepare data to be transmitted: [q_scalar_part, 
    %                                  q_vector_part,
    %                                  r_vector_in_m]
    Global_Chaser_Quaternion_Translation =[sframe_rotation_quat(4,timestep); sframe_rotation_quat(1:3,timestep);Global_Chaser_Translation*0.001];
    %sframe_store(:,timestep)             = Global_Chaser_Quaternion_Translation;

    % Start running the program ont he xpc target:
    if (timestep==1)
        % This launches the target code and records the timestamp.
        % A target model should aready have been compiled and loaded.
        start_time_tg    = tic;
        %tg.start
        
        %return_from_start= tic;
        
        % Record the difference between the tg object being called and
        % returning to the VICON_driver file:
        %time_vector(timestep)   = toc(start_time_tg)    ; %#ok
        %time_vector_2(timestep) = toc(return_from_start); %#ok
    end
    %     Global_Chaser_Quaternion_Translation;
    % TRANSMIT VICON DATA TO TARGET PC THROUGH UDP
    %transm_start         = tic;
    data = mat2str(Global_Chaser_Quaternion_Translation);
    disp(data);
    fwrite(udp_object,data);    
    %transm_log(timestep) = toc(start_time_tg);
    %t = toc(start_time_tg);


end

% Finish execution of the tg object:
%tg.stop

% Close UDP Connection to target PC:
fclose(udp_object);
delete(udp_object);

%------------------------ Tell PC-104 -------------------------------------
if (synch_with_pc104==1)
    fwrite(t,0);  % to stop grabbing images
    fclose(t);    % and close the connection
end
%--------------------------------------------------------------------------

% Disconnect and dispose VICON
if TransmitMulticast
  MyClient.StopTransmittingMulticast();
end 
MyClient.Disconnect();

% Extract data:
fprintf( 'Getting data:\n' );
% out_vicon_EKF = tg.OutputLog;
% tout          = tg.TimeLog;
% yout          = out_vicon_EKF;
% 
% % Save all data contained in the workspace:
% filename=[datestr(now,'yyyymmddTHHMMSS'),'.mat'];
% save(filename)

% Plot results:
%fprintf( 'Plotting data:\n' );
%plot_outputs;
%fprintf( 'Plotting data done\n' );














