package tensor

import "fmt"

func Fused0931FloorSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Log()
	return r, nil
}
func Fused0931FloorSubLogName() string { return "floor_sub_log" }
