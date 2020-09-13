import { Component, AfterViewInit, OnInit, ElementRef } from '@angular/core';
import { ViewChild } from '@angular/core';
import { FilterService } from '../../services/filter/filter.service';
import { FrameService } from '../../services/frame/frame.service';
import { Filter } from '../../models/filter';
import { SockjsMessageService } from 'src/app/services/sockjs-message.service';
import { Subject } from 'rxjs';

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
  processed_context:any;
  public sockServerResponse: string = ''
  private sockServerResponse$:  Subject<any>

  constructor(
    public frameService: FrameService,
    private filterService: FilterService,
    private sms: SockjsMessageService) {
  }

  capture() {
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
    this.sockServerResponse$ = this.frameService.openImageConnection(context, this.filters)
    this.sockServerResponse$.subscribe({
      next: (v) => {
        this.data_raw = JSON.parse(v.data)["image"]
      }
    });
    this.image1.src = this.data_raw;
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
