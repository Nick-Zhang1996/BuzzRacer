
// WARNING: NEVER push high to both MOSFETs on one side, this will create a short
// and burn out the MOSFETs instantly.

// All pins are high enable
#define PORT_POS_UP 9    
#define PORT_POS_DOWN 6 
#define PORT_NEG_UP 11
#define PORT_NEG_DOWN 10

// Timer interrupt based PWM control for H bridge
// Current off-time method: tie both end to GND
// Also RN the car only goes in one direction
// Overhead for ISP is ???? us, as tested

//timer0 -> Arduino millis() and delay()
//timer1 -> Servo lib
//timer2 -> synchronized multi-channel PWM


volatile float onTime = 0.5; // range (0,1], disable pwm for 0
void setup(){
  
  digitalWrite(PORT_POS_UP, LOW);
  digitalWrite(PORT_POS_DOWN,LOW);
  digitalWrite(PORT_NEG_UP, LOW);
  digitalWrite(PORT_NEG_DOWN,LOW);
  
  pinMode(PORT_POS_UP,OUTPUT);
  pinMode(PORT_POS_DOWN,OUTPUT);
  pinMode(PORT_NEG_UP,OUTPUT);
  pinMode(PORT_NEG_DOWN,OUTPUT);
  
  digitalWrite(PORT_NEG_DOWN,HIGH);
  pinMode(13, OUTPUT);
  Serial.begin(9600);
  
  enablePWM();

}//end setup

void enablePWM(){
  cli();//stop interrupts

//set timer2 interrupt 
  TCCR2A = 0;// set entire TCCR2A register to 0
  TCCR2B = 0;// same for TCCR2B
  TCNT2  = 0;//initialize counter value to 0

  // Set CS21 bit for 8 prescaler
  TCCR2B |= (1 << CS21); 
  // duty cycle: (16*10^6) / (8*256) Hz = 7.8kHz
  // set compare target, should update 
  // for n% signal OCR2A = (int) 256*n%
  OCR2A = 128;
  //OCR2A = (uint8_t) 256.0*onTime;    
  // enable timer compare interrupt and overflow interrupt
  TIMSK2 |= (1 << OCIE2A) | ( 1 << TOIE2);


sei();//allow interrupts
  
}

void disablePWM(){
  
  
  cli();//stop interrupts
  //unset timer2 interrupt 
  TCCR2A = 0;// set entire TCCR2A register to 0
  TCCR2B = 0;// same for TCCR2B
  TCNT2  = 0;//initialize counter value to 0
  TIMSK2 = 0;

  sei();//allow interrupts
  
  // TODO maybe tie one end to GND?
  digitalWrite(PORT_POS_UP, LOW);
  digitalWrite(PORT_POS_DOWN,LOW);
  digitalWrite(PORT_NEG_UP, LOW);
  digitalWrite(PORT_NEG_DOWN,LOW);
  
}


// At the falling edge of on-time, enter off-time configuration here
ISR(TIMER2_COMPA_vect){
// TODO: Do this like a pro: write 1 to PINxN to toggle PORTxN
/*
    digitalWrite(PORT_POS_UP, LOW);
    digitalWrite(PORT_NEG_UP, LOW);
    digitalWrite(PORT_POS_DOWN,HIGH);
    digitalWrite(PORT_NEG_DOWN,HIGH);
    digitalWrite(13,LOW);
*/
    
    asm (
      "cbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTB)), "I" (PORTB1)
    );
    
    asm (
      "sbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTD)), "I" (PORTD6)
    );
 
}

// Beginning of each Duty Cycle, enter on-time configuration here
ISR(TIMER2_OVF_vect){
// Do this like a pro: write 1 to PINxN to toggle PORTxN
/*
    digitalWrite(PORT_NEG_UP, LOW);
    digitalWrite(PORT_POS_DOWN,LOW);
    digitalWrite(PORT_POS_UP, HIGH);
    digitalWrite(PORT_NEG_DOWN,HIGH);
    digitalWrite(13,HIGH);
*/
    
    asm (
      "cbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTD)), "I" (PORTD6)
    );
    
    asm (
      "sbi %0, %1 \n"
      : : "I" (_SFR_IO_ADDR(PORTB)), "I" (PORTB1)
    );
}

unsigned long timestamp = 0;
// forward only, range 0-1
#define MAX_H_BRIDGE_POWER 0.2
void setHbridgePower(float power){
    if (power<0.0 || power>1.0){
        disablePWM();
    } else{
        OCR2A = (uint8_t) 256.0*onTime*MAX_H_BRIDGE_POWER;
    }
    return;
}
        
void loop(){
  //do other things here
  if (millis()-timestamp < 500 and false){
    cli();
    onTime = (onTime >=1)?0.01:onTime+0.1;
    OCR2A = (uint8_t) 256.0*onTime;
    sei();
    timestamp = millis();
    Serial.println(OCR2A);
  }
  
}
