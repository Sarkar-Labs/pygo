package tensor

import "fmt"

func Fused0932FloorMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Log()
	return r, nil
}
func Fused0932FloorMulLogName() string { return "floor_mul_log" }
