package tensor

import "fmt"

func Fused0677LogMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Floor()
	return r, nil
}
func Fused0677LogMulFloorName() string { return "log_mul_floor" }
