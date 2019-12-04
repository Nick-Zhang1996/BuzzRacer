// ppm generation, for interfacing with pc without ROS
// Nick Zhang 2019
// The PPM signal is generated with the following convention
// A complete PPM frame is about 22.5 ms (can vary between manufacturer), and signal low state is always 0.3 ms. It begins with a start frame (high state for more than 2 ms). Each channel (up to 8) is encoded by the time of the high state (PPM high state + 0.3 × (PPM low state) = servo PWM pulse width).
// This means a 1000-2000 PWM translates to ???? High state

#include <Arduino.h>

#define CHANNEL_NO 2
#define RC_MAX 2000
#define RC_MIN 1000

const int output_pin[2] = {4,5};
// channels go from 1 - channel_no
// the extra channel 0 is reserved for TSYNC to indicate end of a frame
// unit of value is us for servo PWM
// realistically 
// XXX There may be a race condition on this variable
volatile uint16_t channel[2][CHANNEL_NO+1] = {0};
volatile bool lock_channel = false;

// Timer 1 COMPA would generate rising edge
// Timer 1 COMPB would generate falling edge
void timer1_init(){
    //set timer1 interrupt 
    TCCR1A = 0;// set entire TCCR1A register to 0
    TCCR1B = 0;// same for TCCR1B
    TCCR1C = 0; // PWM related stuff
    TIFR1 |= (1<<TOV1) | (1<<OCF1B) | (1<<OCF1A); // writing 1 to TOV1 clears the flag, preventing the ISR to be activated as soon as sei();
    TCNT1 = 0;

    // prescaler: 256
    // resolution: 1/16e6*256 = 16us (4096us between overflow for 8 bit counter)
    // this starts counting
    TCCR1B |= (1 << CS12) ; 

    // enable timer compare interrupt and overflow interrupt
    TIMSK1 = (1 << OCIE1A) | ( 1 << OCIE1B); 
// for reference, ovf interrupt
    //TIMSK1 = (1 << TOIE1);
}

// Timer 2 COMPA would generate rising edge
// Timer 2 COMPB would generate falling edge
void timer2_init(){
    //set timer2 interrupt 
    TCCR2A = 0;// set entire TCCR2A register to 0
    TCCR2B = 0;// same for TCCR1B
    TIFR2 |= (1<<TOV2) | (1<<OCF2B) | (1<<OCF2A); // writing 1 to TOV1 clears the flag, preventing the ISR to be activated as soon as sei();
    TCNT2 = 0;

    // prescaler: 256
    // resolution: 1/16e6*256 = 16us (4096us between overflow for 8 bit counter)
    // this starts counting
    TCCR2B |= (1 << CS22) | (1 << CS21); 

    // enable timer compare interrupt and overflow interrupt
    TIMSK2 = (1 << OCIE2A) | ( 1 << OCIE2B); 
// for reference, enable ovf interrupt
    //TIMSK2 = (1 << TOIE2);
}
// this will set ISR(TIMER1_COMPA_vect) to fire in specified delay
// the ISR will determine the proper action to take depending on the value in pending_action
void next_action_t1a(float us){

    uint16_t ticks = TCNT1 + (us/16);

    cli();// is this really necessary?
    OCR1A  = ticks;
    sei();

}

void next_action_t1b(float us){

    uint16_t ticks = TCNT1 + (us/16);

    cli();// is this really necessary?
    OCR1B  = ticks;
    sei();

}

// this will set ISR(TIMER2_COMPA_vect) to fire in specified delay
// the ISR will determine the proper action to take depending on the value in pending_action
void next_action_t2a(float us){

    uint16_t ticks = TCNT2 + (us/16);

    cli();// is this really necessary?
    OCR2A  = ticks;
    sei();

}

void next_action_t2b(float us){

    uint16_t ticks = TCNT2 + (us/16);

    cli();// is this really necessary?
    OCR2B  = ticks;
    sei();

}

// raise output at proper timing
volatile uint8_t next_channel[2] = {0};
ISR(TIMER1_COMPA_vect) {
    // TODO assembly this
    digitalWrite(output_pin[0],HIGH);
    next_channel[0]++;
    next_channel[0] %= CHANNEL_NO + 1;

    // assert channel[?] > 300
    if (channel[0][next_channel[0]]<301){
      channel[0][next_channel[0]]=310;
    }
    if (next_channel[0]==0){
        next_action_t1a(4000);
        next_action_t1b(300);
    }else{
        next_action_t1a(channel[0][next_channel[0]]);
        next_action_t1b(300);
    }
}

ISR(TIMER1_COMPB_vect) {
    // TODO assembly this
    digitalWrite(output_pin[0],LOW);
}

// raise output at proper timing
ISR(TIMER2_COMPA_vect) {
    // TODO assembly this
    digitalWrite(output_pin[1],HIGH);
    next_channel[1]++;
    next_channel[1] %= CHANNEL_NO + 1;

    // assert channel[?] > 300
    if (channel[1][next_channel[1]]<301){
      channel[1][next_channel[1]]=310;
    }
    if (next_channel[1]==0){
        next_action_t2a(4000);
        next_action_t2b(300);
    }else{
        next_action_t2a(channel[1][next_channel[1]]);
        next_action_t2b(300);
    }
}

ISR(TIMER2_COMPB_vect) {
    // TODO assembly this
    digitalWrite(output_pin[1],LOW);
}
void setup(){
    // start of the frame
    channel[0][0] = 0;
    channel[0][1] = 1500;
    channel[0][2] = 1500;

    channel[1][0] = 0;
    channel[1][1] = 1500;
    channel[1][2] = 1500;
    pinMode(output_pin[0],OUTPUT);
    pinMode(output_pin[1],OUTPUT);
    pinMode(13,OUTPUT);
    timer1_init();
    timer2_init();
    Serial.begin(115200);
}

char buffer[32];
int c;
int ch1[2],ch2[2];
uint8_t p_buffer=0;
unsigned long ts=0;
bool flag_failsafe=false;

int rc_constrain(int val){
    if (val>RC_MAX){
        return RC_MAX;
    } else if (val<RC_MIN){
        return RC_MIN;
    } else {
        return val;
    }
}

void loop(){
    // command format #1500,1600 \nl
    if (Serial.available()>0) {
        c = Serial.read();
        while (c != -1){
            buffer[p_buffer] = c;
            p_buffer++;
            if (c=='\n'){
                // parse
                buffer[p_buffer+1] = 0;
                ch1[0] = atoi(strtok(buffer,","));
                ch2[0] = atoi(strtok(NULL,",\n"));
                ch1[1] = atoi(strtok(NULL,",\n"));
                ch2[1] = atoi(strtok(NULL,",\n"));
                p_buffer = 0;
                channel[0][1] = rc_constrain(ch1[0]);
                channel[0][2] = rc_constrain(ch2[0]);
                channel[1][1] = rc_constrain(ch1[1]);
                channel[1][2] = rc_constrain(ch2[1]);
                ts = millis();
                flag_failsafe = false;
            }
            
            c = Serial.read();
        }
    }
    // failsafe
    // range limit
    if ((millis()-ts)>500 and !flag_failsafe){
        channel[0][1] = 1500;
        channel[0][2] = 1500;
        channel[1][1] = 1500;
        channel[1][2] = 1500;
        flag_failsafe = true;
    }
    Serial.print(channel[0][1]);
    Serial.print(',');
    Serial.print(channel[0][2]);
    Serial.print(',');
    Serial.print(channel[1][1]);
    Serial.print(',');
    Serial.print(channel[1][2]);
    Serial.print(',');
    Serial.println(flag_failsafe);
}
