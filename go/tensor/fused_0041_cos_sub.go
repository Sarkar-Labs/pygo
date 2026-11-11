package tensor

import "fmt"

func Fused0041CosSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	return r, nil
}
func Fused0041CosSubName() string { return "cos_sub" }
