package tensor

import "fmt"

func Fused1577Log2MulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Floor()
	return r, nil
}
func Fused1577Log2MulFloorName() string { return "log2_mul_floor" }
