package tensor

import "fmt"

func Fused1344GeluMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Log2()
	return r, nil
}
func Fused1344GeluMatmulLog2Name() string { return "gelu_matmul_log2" }
