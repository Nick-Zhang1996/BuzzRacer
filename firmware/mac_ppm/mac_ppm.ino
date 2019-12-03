// ppm generation, for interfacing with pc without ROS
// Nick Zhang 2019
// The PPM signal is generated with the following convention
// A complete PPM frame is about 22.5 ms (can vary between manufacturer), and signal low state is always 0.3 ms. It begins with a start frame (high state for more than 2 ms). Each channel (up to 8) is encoded by the time of the high state (PPM high state + 0.3 × (PPM low state) = servo PWM pulse width).
// This means a 1000-2000 PWM translates to ???? High state

#include <Arduino.h>

#define CHANNEL_NO 4
#define RC_MAX 2000
#define RC_MIN 1000

const int output_pin = 4;
// channels go from 1 - channel_no
// the extra channel 0 is reserved for TSYNC to indicate end of a frame
// unit of value is us for servo PWM
// realistically 
// XXX There may be a race condition on this variable
volatile uint16_t channel[CHANNEL_NO+1] = {0};
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

    // prescaler: 64
    // duty cycle: (16*10^6) / (64*65536) Hz = 38Hz (3300us between overflow)
    // overflow interval (64*65536)/(16e6) = 26ms 
    // per count(resolution) : 0.4us
    // this starts counting
    TCCR1B |= (1 << CS11) | (1 << CS10) ; 

    // enable timer compare interrupt and overflow interrupt
    TIMSK1 = (1 << OCIE1A) | ( 1 << OCIE1B); 
// for reference, ovf interrupt
    //TIMSK1 = (1 << TOIE1);
}

// this will set ISR(TIMER1_COMPA_vect) to fire in specified delay
// the ISR will determine the proper action to take depending on the value in pending_action
void next_action_t1a(float us){

    uint16_t ticks = TCNT1 + (us/4);

    cli();// is this really necessary?
    OCR1A  = ticks;
    sei();

}

void next_action_t1b(float us){

    uint16_t ticks = TCNT1 + (us/4);

    cli();// is this really necessary?
    OCR1B  = ticks;
    sei();

}

// raise output at proper timing
volatile uint8_t next_channel = 0;
ISR(TIMER1_COMPA_vect) {
    // TODO assembly this
    digitalWrite(output_pin,HIGH);
    next_channel++;
    next_channel %= CHANNEL_NO + 1;

    // assert channel[?] > 300
    if (channel[next_channel]<301){
      channel[next_channel]=310;
    }
    if (next_channel==0){
        next_action_t1a(6000);
        next_action_t1b(300);
    }else{
        next_action_t1a(channel[next_channel]);
        next_action_t1b(300);
    }
}

ISR(TIMER1_COMPB_vect) {
    // TODO assembly this
    digitalWrite(output_pin,LOW);
}

void setup(){
    // start of the frame
    channel[0] = 0;

    channel[1] = 1500;
    channel[2] = 1500;
    channel[3] = 1500;
    channel[4] = 1500;
    pinMode(output_pin,OUTPUT);
    pinMode(13,OUTPUT);
    timer1_init();
    Serial.begin(115200);
}

char buffer[32];
int c;
int ch1,ch2;
uint8_t p_buffer=0;
unsigned long ts=0;
bool flag_failsafe=false;
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
                ch1 = atoi(strtok(buffer,","));
                ch2 = atoi(strtok(NULL,",\n"));
                p_buffer = 0;
                if (ch1>RC_MAX){
                    channel[1] = RC_MAX;
                } else if (ch1<RC_MIN){
                    channel[1] = RC_MIN;
                } else {
                    channel[1] = ch1;
                }
                if (ch2>RC_MAX){
                    channel[2] = RC_MAX;
                } else if (ch2<RC_MIN){
                    channel[2] = RC_MIN;
                } else {
                    channel[2] = ch2;
                }
                ts = millis();
                flag_failsafe = false;
            }
            
            c = Serial.read();
        }
    }
    // failsafe
    // range limit
    if ((millis()-ts)>500 and !flag_failsafe){
        channel[1] = 1500;
        channel[2] = 1500;
        flag_failsafe = true;
    }
    //Serial.print(channel[1]);
    //Serial.print(',');
    //Serial.print(channel[2]);
    //Serial.print(',');
    //Serial.println(flag_failsafe);
}
