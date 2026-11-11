package tensor

import "fmt"

func Fused0675LogAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Floor()
	return r, nil
}
func Fused0675LogAddFloorName() string { return "log_add_floor" }
