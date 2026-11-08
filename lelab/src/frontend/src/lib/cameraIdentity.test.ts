import { expect, test } from 'vitest';
import { matchAvailableCamera } from './cameraConfig';

const live=[{index:4,deviceId:'wsl-camera:head',role:'head'},
            {index:0,deviceId:'wsl-camera:wrist',role:'wrist'}];
test('legacy index bindings migrate by camera role after swapped node numbers', () => {
  expect(matchAvailableCamera({name:'head',device_id:'wsl-camera:0',camera_index:0},live)?.index).toBe(4);
});
test('missing stable and named cameras do not bind to the other camera at an old index', () => {
  expect(matchAvailableCamera({name:'head',device_id:'wsl-camera:head',camera_index:0},[live[1]])).toBeUndefined();
  expect(matchAvailableCamera({name:'head',device_id:'wsl-camera:0',camera_index:0},[live[1]])).toBeUndefined();
});
