import { Component, AfterViewInit, OnInit, ElementRef } from '@angular/core';
import { ViewChild } from '@angular/core';
import { FilterService } from '../../services/filter/filter.service';
import { FrameService } from '../../services/frame/frame.service';
import { Filter } from '../../models/filter';
import { SockjsMessageService } from 'src/app/services/sockjs-message.service';
import { Observer, Observable, throwError, Subject } from 'rxjs';

@Component({
  selector: 'app-display',
  templateUrl: './display.component.html',
  styleUrls: ['./display.component.css']
})
export class DisplayComponent implements AfterViewInit, OnInit {
  @ViewChild('videoElement') videoElement: ElementRef;
  @ViewChild('unprocessedCanvas') unprocessedCanvas: ElementRef;
  @ViewChild('processedCanvas') processedCanvas: ElementRef;
  data_raw: any;

  filters: Filter[];
  isPlaying = false;
  displayControls = true;
  image: any;
  image1: any;
  processed_context: any;
  lastTime: any;
  readyToSend = true;
  public sockServerResponse: string = ''
  private sockServerResponse$:  Subject<any>
  sock: any;

  constructor(
    public frameService: FrameService,
    private filterService: FilterService,
    private sms: SockjsMessageService) {
  }

  capture() {
    requestAnimationFrame(this.capture.bind(this))
    var time = this.videoElement.nativeElement.currentTime;
    if (time == 0) {
      return
    }
    if (time == this.lastTime) {
        // console.log('time: ' + time);
        //todo: do your rendering here
        return;
    }
    this.lastTime = time;

    if (this.unprocessedCanvas.nativeElement.width!=this.videoElement.nativeElement.clientWidth)
      this.unprocessedCanvas.nativeElement.width = this.videoElement.nativeElement.clientWidth;
    if (this.unprocessedCanvas.nativeElement.height!=this.videoElement.nativeElement.clientHeight)
      this.unprocessedCanvas.nativeElement.height = this.videoElement.nativeElement.clientHeight;
    var context = this.unprocessedCanvas.nativeElement.getContext("2d");
    context.drawImage(this.videoElement.nativeElement, 0, 0, this.videoElement.nativeElement.videoWidth, this.videoElement.nativeElement.videoHeight, 0, 0, this.videoElement.nativeElement.clientWidth, this.videoElement.nativeElement.clientHeight);
    this.processFrame(this.unprocessedCanvas.nativeElement.toDataURL());

  }

  ngOnInit(): void {
    this.getFilters();
    this.sock = this.frameService.openImageConnection()
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  // openSockConn() {
  //   this.sockServerResponse$ = this.sms.openImageConnection('bla')
  //   this.sockServerResponse$.subscribe({
  //     next: (v) => {
  //       this.sockServerResponse = v.data
  //       console.log(JSON.stringify(v))
  //     }
  //   })
  // }

  processFrame(context): any {
    if (this.processedCanvas.nativeElement.width!=this.videoElement.nativeElement.clientWidth)
      this.processedCanvas.nativeElement.width = this.videoElement.nativeElement.clientWidth;
    if(this.processedCanvas.nativeElement.height != this.videoElement.nativeElement.clientHeight)
      this.processedCanvas.nativeElement.height = this.videoElement.nativeElement.clientHeight;
    this.image1 = new Image();
    this.image1.onload = ()=> {
      var processed_context = this.processedCanvas.nativeElement.getContext("2d");
      processed_context.drawImage(this.image1, 0, 0, this.videoElement.nativeElement.clientWidth, this.videoElement.nativeElement.clientHeight, 0, 0, this.videoElement.nativeElement.clientWidth, this.videoElement.nativeElement.clientHeight);
    }
    let wsjs = this.sock;
    let frame = context;
    let config = this.filters;
    let observable = new Observable((obs:Observer<MessageEvent>) => {
      wsjs.onmessage = obs.next.bind(obs)
      wsjs.onerror = obs.error.bind(obs)
      wsjs.onclose = obs.complete.bind(obs)
      wsjs.onopen = () =>{
        console.log('open')
        // let sendObject = {
        //   path: this.setImage,
        //   slika: imageEncoded
        // }

        // wsjs.send(JSON.stringify(sendObject));
        let post_data = {'frame': frame, 'config': config}
        wsjs.send(JSON.stringify(post_data))
      }
    })

    let subject = new Subject()
    subject.subscribe({
      next: (v) => {
        // this.data_raw = JSON.parse(v.data)["image"]
        console.log(v)
      },
      error: (v) => {
        console.log(v)
      }
    });
    observable.subscribe(subject)


    let post_data = {'frame': frame, 'config': config}
    wsjs.onmessage = (v) =>  {
      console.log('Oh snap');
      this.data_raw=JSON.parse(v.data)["image"];
    }
    wsjs.send(JSON.stringify(post_data))
    this.image1.src = this.data_raw;
    // requestAnimationFrame(this.capture.bind(this))
  }


  ngAfterViewInit(): void {
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
          this.videoElement.nativeElement.srcObject = stream;
      });
    }
  }

  start() {
    this.initCamera({ video: true, audio: false });
  }

  sound() {
    this.initCamera({ video: true, audio: true });
  }

  pause() {
    this.videoElement.nativeElement.pause();
  }

  toggleControls() {
    this.videoElement.nativeElement.controls = this.displayControls;
    this.displayControls = !this.displayControls;
  }

  resume() {
    this.videoElement.nativeElement.play();
  }

  initCamera(config:any) {
    var browser = <any>navigator;

    browser.getUserMedia = (browser.getUserMedia ||
      browser.webkitGetUserMedia ||
      browser.mozGetUserMedia ||
      browser.msGetUserMedia);

    browser.mediaDevices.getUserMedia(config).then(stream => {
      this.videoElement.nativeElement.srcObject = stream;
      this.videoElement.nativeElement.play();
    });
  }

  openSockConn() {
    this.sockServerResponse$ = this.sms.openImageConnection('bla')
    this.sockServerResponse$.subscribe({
      next: (v) => {
        this.sockServerResponse = v.data
        console.log(JSON.stringify(v))
      }
    })
  }
}
