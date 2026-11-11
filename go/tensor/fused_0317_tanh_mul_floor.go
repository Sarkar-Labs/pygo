package tensor

import "fmt"

func Fused0317TanhMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Floor()
	return r, nil
}
func Fused0317TanhMulFloorName() string { return "tanh_mul_floor" }
