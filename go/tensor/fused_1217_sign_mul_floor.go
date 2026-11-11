package tensor

import "fmt"

func Fused1217SignMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Floor()
	return r, nil
}
func Fused1217SignMulFloorName() string { return "sign_mul_floor" }
