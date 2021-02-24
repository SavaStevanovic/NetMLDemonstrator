import { Component, OnInit } from '@angular/core';
import { StateService } from '../../services/state/state.service'
import { Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-footer',
  templateUrl: './footer.component.html',
  styleUrls: ['./footer.component.css']
})
export class FooterComponent implements OnInit {
  videoPlaying: boolean;
  quality: number;
  togleMenu = true;

  constructor(private stateService: StateService) {
  }

  ngOnInit(): void {
    this.stateService.videoPlaying$
      .subscribe(videoPlaying => this.videoPlaying = videoPlaying);
    this.quality = 0.5;
  }

  updateQuality(): void {
    this.stateService.setVideoQuality(this.quality);
  }

  @Output() menuToggled = new EventEmitter();
  toggleMenu(): void {
    this.menuToggled.emit();
  }

  pause(): void {
    this.stateService.setVideoStart(false);
  }

  resume(): void {
    this.stateService.setVideoStart(true);
  }
}
