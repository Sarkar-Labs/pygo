package tensor

import "fmt"

func Fused1669Log10MatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Floor()
	return r, nil
}
func Fused1669Log10MatmulFloorName() string { return "log10_matmul_floor" }
