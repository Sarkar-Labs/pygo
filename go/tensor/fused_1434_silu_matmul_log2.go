package tensor

import "fmt"

func Fused1434SiluMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Log2()
	return r, nil
}
func Fused1434SiluMatmulLog2Name() string { return "silu_matmul_log2" }
